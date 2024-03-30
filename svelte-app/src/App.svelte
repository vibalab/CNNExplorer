<script>
	import { onMount, tick } from 'svelte';
  import * as d3 from 'd3';
  import { Input, FormGroup, Label, FormCheck, Button, Row, Col, Modal, ModalBody, ModalHeader, ModalFooter } from 'sveltestrap';

  //######################################################################//
  let selectedModel = undefined;
  let selectedClass = undefined;
  let selectedModule = undefined;
  let selectedBranch = undefined;
  let moduleLayerDepth = undefined;
  let modelSVG = undefined;
  let imageNum = 8;
  let pathData = [];
  let link = undefined;

  const branches = ['branch1','branch2','branch3','branch4'];
  const imagenetModels = ['alexnet', 'vgg16', 'googlenet', 'resnet18'];
  let imagenetClasses ={}
  onMount(async () => {
    const response = await fetch('/imageClasses.json');
    imagenetClasses = await response.json();
    selectedModel = imagenetModels[0];
    selectedClass = "0";
    modelSVG = d3.select('#model-load')
                    .append('svg')
                    .attr('width', 2000)
                    .attr('height', 1000);  
    link = d3.linkHorizontal()
      .x(d=>d[0])
      .y(d=>d[1]);
  });

  let modelData = undefined;
  const moduleXMargin = 20;
  const moduleYMargin = 200;
  const moduelWidth = 100;
  const moduleHeight = 400;
  const moduleStruct = {
    'alexnet':['conv','conv','conv','avgpool','linear'],
    'vgg16': ['conv','conv','conv','conv','conv','avgpool','linear'],
    'googlenet':['conv','conv','inception','inception','inception','inception','inception','inception','inception','inception','inception','avgpool','linear'],
    'resnet18':['conv','residual','residual','residual','residual','residual','residual','residual','residual','avgpool','linear']
  };

  let openModal = false;
  let batchNormActive = false;
  let reluActive = false;
  const imageHeight = 133;
  const imageWidth = 133;
  const moduleXPadding = 30;
  const moduleYPadding = 30;
  const offsetX = 100;
  const offsetY = 30;

  function updateSVGSize(newWidth, newHeight) {
    d3.select('#detail-svg')
      .attr('width', newWidth)
      .attr('height', newHeight);

    // const svg = d3.select('#detail-svg');
    // const container = d3.select('#svg-container');

    // // SVG의 크기를 구하고 컨테이너의 스타일을 업데이트
    // const width = svg.getBBox().width;
    // const height = svg.getBBox().height;

    // container.style.width = `${newWidth}px`;
    // container.style.height = `${height}px`;
  }

  async function loadModelView() {
    modelSVG.selectAll('*').remove();
    const response = await fetch(`/output/${selectedClass}/${selectedModel}_info.json`);
    modelData = await response.json();
    console.log("Loaded JSON data:", modelData);

    // JSON 객체의 모든 키를 출력
    console.log("Keys in JSON:", Object.keys(modelData));

    const moduleGroup = modelSVG.append('g').attr('class', 'module-group');

    const modules = moduleGroup.selectAll('g')
      .data(moduleStruct[selectedModel])
      .enter()
      .append('g')
      .attr('class', 'module')
      .attr('transform', (d, i) => `translate(${i * (moduelWidth + moduleXMargin)}, ${moduleYMargin})`);

    // 각 하위 g 요소 안에 rect 추가
    modules.append('rect')
      .attr('width', moduelWidth)
      .attr('height', moduleHeight)
      .style('fill', (d) => moduleFills(d))
      .style('stroke', 'gray')
      .style('stroke-width', 0);

    // 각 하위 g 요소 안에 text 추가
    modules.append('text')
      .attr('x', moduelWidth / 2)
      .attr('y', moduleHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .text((d) => d)
      .style('fill', 'black');

      const expandedWidth = moduelWidth * 2; // 확장할 너비
      const shiftDistance = expandedWidth - moduelWidth; // 확장으로 인해 밀어낼 거리

      modules.each(function(d, i) {
        const group = d3.select(this);

        group.select('rect')
          .on('mouseover', function() {
            // Expand rect 
            d3.select(this)
              .transition()
              .duration(500)
              .attr('width', expandedWidth)
              .style('stroke-width', 3);

            // 현재 rect 오른쪽의 모든 rect를 오른쪽으로 밀어냄
            modules.filter((_, j) => j > i)
              .transition()
              .duration(500)
              .attr('transform', (d, j) => `translate(${((j + i + 1) * (moduelWidth + moduleXMargin)) + shiftDistance}, ${moduleYMargin})`);

            // Hide text
            group.select('text')
              .transition()
              .duration(500)
              .style('opacity', 0);
          })
          .on('mouseout', function() {
            // 모든 rect를 원래 크기로 복원
            d3.select(this)
              .transition()
              .duration(500)
              .attr('width', moduelWidth)
              .style('stroke-width', 1);

            // 모든 rect를 원래 위치로 복원
            modules.transition()
              .duration(500)
              .attr('transform', (d, j) => `translate(${j * (moduelWidth + moduleXMargin)}, ${moduleYMargin})`);

            // 모든 text를 다시 표시
            modules.select('text')
              .transition()
              .duration(500)
              .style('opacity', 1);
        })
        //Click Effect => Call 'showDetailView'
        .on('click', () => { 
          openDetailView(d, i); 
        });
      });


    function moduleFills(moduleName){
      let moduleFills = undefined;
      if (moduleName === 'conv'){
        moduleFills = 'green'
      }
      else if (moduleName === 'residual'){
        moduleFills = 'red'
      }
      else if (moduleName === 'avgpool'){
        moduleFills = 'yellow'
      }
      else if (moduleName === 'linear'){
        moduleFills = 'orange'
      }
      else if (moduleName === 'inception'){
        moduleFills = 'gray'
      }
      return moduleFills
    }
  }

  // Open Detail View of Selected Module
  async function openDetailView(selectedModuleName, selectedModuleIndex) {
    selectedModule = selectedModuleName;
    openModal = true;
    await tick();

    /* Extract selected module's layers from modelData
      modelData contains all layer information in json format */ 
    const selectedModuleLayers = [];
    const selectedModuleLayerNames = [];
    let moduleIndex = -1;
    let inputLayer = undefined;
    for (const key in modelData) {
      if(modelData[key]['layer_index'] === 0){
        moduleIndex++;
      }
      if(moduleIndex === selectedModuleIndex){
        selectedModuleLayers.push(modelData[key]);
        selectedModuleLayerNames.push(key);
      }
      else if(moduleIndex < selectedModuleIndex){
        inputLayer = modelData[key]['output']
      }
    }
    drawModuleDetail(selectedModuleName, selectedModuleLayers, inputLayer, selectedModuleLayerNames);
  }
  // Close Detail View 
  function closeDetailView() {
    openModal = false;  
    reluActive = false;
    batchNormActive = false;
    selectedBranch = undefined;
    pathData = [];
    d3.select('#detail-svg').selectAll("*").remove();
  }

  // Call drawModule functions depending on the type of module
  function drawModuleDetail(moduleName, moduleLayers, inputLayer, layerNames) {
    if (moduleName === 'conv'){
        drawConvModuleDetail(moduleLayers);
    }
    else if (moduleName === 'avgpool'){
      drawAvgpoolModuleDetail(moduleLayers);
    }
    else if (moduleName === 'linear'){
      drawLinearModuleDetail(moduleLayers, inputLayer);
    }
    else if (moduleName === 'residual'){
      drawResidualModuleDetail(moduleLayers, layerNames);
    }
    else if (moduleName === 'inception'){
      drawInceptionModuleDetail(moduleLayers, layerNames);
    }
    drawLayerConnections();
  }

  function drawLayerConnections(){
    console.log(`depth is ${moduleLayerDepth}`);
    for(let cursor = moduleLayerDepth; cursor > 0; cursor--){
      const currLayer = d3.select('#detail-svg').selectAll('g').filter(function() { 
        if(!this.getAttribute('class').includes('IntermediateResult')){
          return false;
        }
        const isCurrentLayerDepth = this.getAttribute('id').split('-')[1] === String(cursor);
        const isDisplayInline = window.getComputedStyle(this).display === 'inline';
        //In case of inception --> check isInlineBranch
        if(this.getAttribute('class').includes('branch')){
          // const isInlineBranch = this.className.baseVal.includes(selectedBranch);
          const isInlineBranch = this.getAttribute('class').includes(selectedBranch);
          return isDisplayInline && isInlineBranch && isCurrentLayerDepth;
        }
        return isDisplayInline && isCurrentLayerDepth;
    });

    const prevLayer = d3.select('#detail-svg').selectAll('g').filter(function() { 
      if(!this.getAttribute('class').includes('IntermediateResult')){
          return false;
      }
      const isCurrentLayerDepth = this.getAttribute('id').split('-')[1] === String(cursor - 1);
      const isDisplayInline = window.getComputedStyle(this).display === 'inline';
      //In case of inception --> check isInlineBranch
      if(this.getAttribute('class').includes('branch')){
        // const isInlineBranch = this.className.baseVal.includes(selectedBranch);
        const isInlineBranch = this.getAttribute('class').includes(selectedBranch);
        return isDisplayInline && isInlineBranch && isCurrentLayerDepth;
      }
      return isDisplayInline && isCurrentLayerDepth;
    });

      currLayer.each(function() {
        const dstIR = d3.select(this);
        const layerClass = dstIR.attr('class');
        const dstImageIndex = dstIR.attr('id').split('-')[3];

        if (layerClass.includes('IntermediateResult') && layerClass.includes('Conv2d')) {
          prevLayer.each(function() {
            const srcIR = d3.select(this);
            const srcImageIndex = srcIR.attr('id').split('-')[3];
            addHorizontalConnection(srcIR, dstIR, cursor, srcImageIndex ,dstImageIndex);
          });
        }
        else if(layerClass === 'IntermediateResult-Concat'|| layerClass === 'IntermediateResult-Residual' || layerClass.includes('Pool')){  //ToDo(YSKIM): residual -> Add 
          const srcIR = prevLayer.filter(function() {
            return d3.select(this).attr('id').split('-')[3] === dstImageIndex;
          })
          addHorizontalConnection(srcIR, dstIR, cursor, dstImageIndex ,dstImageIndex);
        }
      });
    }
    pathData.forEach(path => {
      d3.select('#detail-svg').append('path')
          .attr('d', link({
            source: path.source,
            target: path.target
          }))
          .attr('fill', 'none')
          .attr('stroke', 'gray')
          .attr('stroke-width', 1)
          .attr('id', path.id)
          .style('stroke-opacity', 0.5);
    });
  }

  function addHorizontalConnection(srcImage, dstImage, dstLayerIndex, srcImageIndex, dstImageIndex){
    const srcTranslate = srcImage.attr('transform').match(/translate\(([^)]+)\)/);
    let srcCoordinates = srcTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    srcCoordinates[0] = srcCoordinates[0] + imageWidth;
    srcCoordinates[1] = srcCoordinates[1] + imageHeight/2;

    const dstTranslate = dstImage.attr('transform').match(/translate\(([^)]+)\)/);
    let dstCoordinates = dstTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    dstCoordinates[1] = dstCoordinates[1] + imageHeight/2;

    pathData.push({ id:`edge-${dstLayerIndex}-${srcImageIndex}-${dstImageIndex}`, source: srcCoordinates, target: dstCoordinates}) 
  }
  // Draw Conv Module 
  function drawConvModuleDetail(moduleLayers){
    let visibleLayerIndex = 0;
    let hiddenLayerCount = 0;
    let x;
    let y;
    moduleLayers.forEach((layer, layerIndex) => {
      //Input Layer
      if(layerIndex === 0){   
        const inputX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const inputY = moduleYPadding + (offsetY);

        drawLayer(layer['input'], visibleLayerIndex, hiddenLayerCount, inputX, inputY, 'inline', 'input');

        visibleLayerIndex++;
        x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        y = moduleYPadding + (offsetY);
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class']);

      }
      //ReLU & BatchNorm Layer
      else if(layer['class'] === 'ReLU' || layer['class'] === 'BatchNorm2d'){
        hiddenLayerCount++;
        x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        y = moduleYPadding + (offsetY)
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['class']);
      }
      //Other Layers (Conv, Pool)
      else if(layer['class'] === 'Conv2d' || layer['class'].includes('Pool')){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        y = moduleYPadding + (offsetY);
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class']);
      }
    });
    moduleLayerDepth = visibleLayerIndex;
  }

  // Draw Avgpool Module
  function drawAvgpoolModuleDetail(moduleLayers){
    moduleLayers.forEach((layer, layerIndex) => {
      //Input Layer 
      if(layerIndex === 0){

        const x = moduleXPadding + (layerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        console.log(`layerIndex: ${layerIndex}`)

        drawLayer(layer['input'], 0, 0, x, y, 'inline', 'input');
      }
      // Other Layers (Pool)
      const x = moduleXPadding + (layerIndex + 1) * (imageWidth + offsetX);
      const y = moduleYPadding + (offsetY)
      drawLayer(layer['output'], layerIndex + 1, layerIndex, x, y, 'inline', layer['class']);
      moduleLayerDepth = layerIndex + 1;
    });
  }
  
  // Draw Linear Module
  function drawLinearModuleDetail(moduleLayers, inputLayer){
    let visibleLayerIndex = 0;
    let hiddenLayerCount = 0;
    console.log(moduleLayers)
    moduleLayers.forEach((layer, layerIndex) => {
      //Input Layer (Original input, Flatten input)
      if(layerIndex === 0){
        const inputX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const inputY = moduleYPadding + (offsetY);
        drawLayer(inputLayer, visibleLayerIndex, hiddenLayerCount, inputX, inputY, 'inline', 'input');

        const x = moduleXPadding + (visibleLayerIndex + 1) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawFlatten3D(inputLayer, 0, 0, x, y, 'inline', 'input');
      }
      //Last Layer contains Top-10 prediction labes (output_index) and probability (output)
      if(layerIndex === (moduleLayers.length - 1)){
        //TODO(YSKIM): Print top 10 labels
        // const x = moduleXPadding + (layerIndex - hiddenLayerCount + 1) * (imageWidth + offsetX);
        // const y = moduleYPadding + (offsetY);
        //drawPredcition(layer, x, y, 1);
      }
      //ReLU Layer
      else if(layer['class'] === 'ReLU'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex + 1) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLinear(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['class']);
      }
      //Other Layers (Linear)
      else if(layer['class'] === 'Linear'){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex + 1) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLinear(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class']);
      }
    });
    moduleLayerDepth = visibleLayerIndex;
  }

  // Draw Residual Module
  function drawResidualModuleDetail(moduleLayers, layerNames){  
    let visibleLayerIndex = 0;
    let hiddenLayerCount = 0;
    const residualY = imageHeight;
    console.log(layerNames)
    console.log(moduleLayers)
    moduleLayers.forEach((layer, layerIndex) => {
      //last RelU Layer includes identity
      if(layerIndex === (moduleLayers.length - 1)){   
        const inputX = moduleXPadding;
        const inputY = moduleYPadding + (offsetY) + residualY;
        drawLayer(layer['identity'], 0, 0, inputX, inputY, 'inline', 'input');

        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY) + residualY;
        const x1 = inputX + imageWidth/2;
        const x2 = x + imageWidth/2;
        drawShortcut(x1, x2, y, 1);
        drawLayer(layer['input'], visibleLayerIndex, 0, x, y, 'inline', 'Residual');
        drawLayer(layer['output'], visibleLayerIndex, 1, x, y, 'none', layer['class']);
      }
      else if(layerNames[layerIndex].includes('downsample')){
        const x1 = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y1 = moduleYPadding + (offsetY) + residualY;

        // const x2 = 
        // cosnt y2 = 

        // drawLayer(layer['identity'], x1, x2, y1, 1);
      } 
      else if(layer['class'] === 'ReLU' || layer['class'] === 'BatchNorm2d'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY) + residualY;

        visibleLayerIndex = visibleLayerIndex;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['class']);
      }
      else if(layer['class'] === 'Conv2d' || layer['class'].includes('Pool')){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY) + residualY;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class']);
      }
    });
    moduleLayerDepth = visibleLayerIndex;

    function drawShortcut(x1, x2, y0, z){
      const detailSVG = d3.select('#detail-svg');
      const stepBeforeLine = d3.line()
                            .x(d => d[0])
                            .y(d => d[1])
                            .curve(d3.curveStepBefore);
      const stepAfterLine = d3.line()
                            .x(d => d[0])
                            .y(d => d[1])
                            .curve(d3.curveStepAfter);
      let y1;
      let y2;
      for(let i = 0; i < imageNum; i++){
        const shortcutGroup = detailSVG.append('g')
                              .attr('class','shortcut')
                              .attr('id', `edge-${i}`);
        y1 = y0 + i * (imageHeight + offsetY);
        y2 = i < imageNum / 2 ? y0 - (imageHeight + offsetY) : y0 + (imageNum + 1) * (imageHeight + offsetY) + imageHeight;

        const pathDataBeforeLine = [
          [x1, y1],
          [(x1 + x2) / 2, y2]
        ];

        const pathDataAfterLine = [
          [(x1 + x2) / 2, y2],
          [x2, y1]
        ];
          
        shortcutGroup.append('path')
          .attr('d', stepBeforeLine(pathDataBeforeLine))
          .attr('fill', 'none')
          .attr('stroke', 'black')
          .attr('class',`line-${i}`)
          .attr('stroke-width', 1)
          .style('stroke-opacity', 0);
          
        shortcutGroup.append('path')
          .attr('d', stepAfterLine(pathDataAfterLine))
          .attr('fill', 'none')
          .attr('stroke', 'black')
          .attr('class',`line-${i}`)
          .attr('stroke-width', 1)
          .style('stroke-opacity', 0);
      }
      detailSVG.selectAll('path.line-0').style('stroke-opacity', 1)
      updateSVGSize(x2 + imageWidth, y2);
    }
    function drawDownSample(layer, x1, y1, x2, y2, z){
  
    }
  }

  // Draw Inception Module
  function drawInceptionModuleDetail(moduleLayers, layerNames){
    console.log(moduleLayers);
    console.log(layerNames);
    // branchLayerNum = getBranchStruct()
    let branchName = '';
    let hiddenLayerCount = 0;
    let visibleLayerIndex = 0;
    const detailSVG = d3.select('#detail-svg');
    let brachGroup = undefined;

    moduleLayers.forEach((layer, layerIndex) => {
      let currentBranch = layerNames[layerIndex].split('.')[1];
      //Input Layer 
      if(layerIndex === 0){
        const x = moduleXPadding + (layerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding;

        drawLayer(layer['input'], visibleLayerIndex, 0, x, y, 'inline', 'Input');
      }
      //Last Layer
      if(layerIndex === moduleLayers.length - 1){
        const x = moduleXPadding + 3 * (imageWidth + offsetX);
        const y = moduleYPadding;
        //Draw output layer
        moduleLayerDepth = 2;
        drawLayer(layer['output'], moduleLayerDepth, 0, x, y, 'inline', 'Concat'); 
      }
      if(branchName !== currentBranch){
        hiddenLayerCount = 0;
        visibleLayerIndex = 0;
        branchName = currentBranch
        brachGroup = detailSVG.append('g')
                              .attr('id', `${branchName}`)
                              .attr('class', 'branch');
      }
      if(layer['class'] === 'ReLU' || layer['class'] === 'BatchNorm2d'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['class'], branchName);
      }
      //Other Layers (Conv, Pool)
      else if(layer['class'] === 'Conv2d' || layer['class'].includes('Pool')){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class'], branchName);
      }
    });
    selectedBranch = 'branch1';
  }

  function updateInceptionBranch(){
    if(typeof selectedBranch !== 'undefined'){
      console.log('updateInception')
      //visible인거 전부 hidden으로
      d3.selectAll('g.branch')
                      .style('display', 'none');

      //hidden중에 selectedBranch인거 show하기
      d3.select(`g#${selectedBranch}`)
                      .style('display', 'inline');

      const numLayerCurrentBranch = d3.select(`g#${selectedBranch}`)
                                    .selectAll('g')
                                    .filter(function() {return window.getComputedStyle(this).display === 'inline';})
                                    .size() / 8;

      d3.select('#detail-svg').selectAll('g.IntermediateResult-Concat').each(function(){
        const currentImageIndex = d3.select(this).attr('id').split('-')[3];
        const newImageIndex = `IR-${numLayerCurrentBranch + 1}-0-${currentImageIndex}`
        d3.select(this).attr('id', newImageIndex);
      });

      moduleLayerDepth = numLayerCurrentBranch + 1;

      d3.select('#detail-svg').selectAll('path').remove();
      pathData = [];
      drawLayerConnections();
    }
  }

  $: if(selectedBranch) {
    updateInceptionBranch();
  }

  function drawFlatten3D(layer, visibleLayerIndex, layerIndex, x, y, display = 'inline', layerClass){
    const flatImages = layer.flat(3);
    drawLinear(flatImages, visibleLayerIndex, layerIndex, x, y, display, layerClass);
  }

  function drawLinear(layer, visibleLayerIndex, layerIndex, x, y, display = 'inline', layerClass){
    const [max, min] = getLayerMaxMin(layer);
    const boundaryValue = Math.max(Math.abs(min), Math.abs(max));
    const colorScale = d3.scaleLinear()
    .domain([-boundaryValue, boundaryValue])
    .range(['red', 'blue']); 

    const detailSVG = d3.select('#detail-svg');
    const linearLayerGroup = detailSVG.append('g')
                              .attr('class', `IntermediateResult-${layerClass}`)
                              .attr('id', `IR-${visibleLayerIndex}-${layerIndex}-0`)
                              .attr('transform', `translate(${x}, ${y})`)
                              .style('display', display);

    const layerHeight = (imageHeight + offsetY) * imageNum - offsetY;
    const layerWidth = 40;
    const linearRectHeight = layerHeight / layer.length;
    const linearRectWidth = layerWidth;

    layer.forEach((value, index) => {
      linearLayerGroup.append('rect')
        .attr('x', 0)
        .attr('y', linearRectHeight * index)
        .attr('width', linearRectWidth)
        .attr('height', linearRectHeight)
        .attr('id', `block-${index}`)
        .style('fill', colorScale(value))
        .style('stroke', 'black')
        .style('stroke-opacity', 0.5);
    });

    const currentWidth = detailSVG.attr('width');
    const currentHeight = detailSVG.attr('height');
    const requiredWidth = Math.max(currentWidth, x + layerWidth);
    const requiredHeight = Math.max(currentHeight, y + layerHeight);

    updateSVGSize(requiredWidth, requiredHeight);
  }

  // Draw a layer which is contains IR results (Image or Linear)
  function drawLayer(layerImages, visibleLayerIndex, layerIndex, layerX, layerY, display, layerClass, branchName = 'none'){
    // Color Sacle of the layer images
    const [max, min] = getLayerMaxMin3D(layerImages);
    const boundaryValue = Math.max(Math.abs(min), Math.abs(max));
    const colorScale = d3.scaleLinear()
    .domain([-boundaryValue, boundaryValue])
    .range(['red', 'blue']); 

    let ImageX = 0;
    let ImageY = 0;
    layerImages.forEach((image, imageIndex) => {
      ImageX = layerX;
      ImageY = layerY + (imageHeight + offsetY) * (imageIndex);
      drawImage(image, visibleLayerIndex, layerIndex, imageIndex, colorScale, ImageX, ImageY, display, layerClass, branchName);
    });


    // If layers over the svg size, update svg size.
    const detailSVG = d3.select('#detail-svg');
    const imageSize = 133;
    const currentWidth = detailSVG.attr('width');
    const currentHeight = detailSVG.attr('height');
    const requiredWidth = Math.max(currentWidth, ImageX + imageSize);
    const requiredHeight = Math.max(currentHeight, ImageY + imageSize);

    updateSVGSize(requiredWidth, requiredHeight);

    //TODO(YSKIM): Make Legend 
  }


  //Draw an Image
  function drawImage(image, visibleLayerIndex, layerIndex, imageIndex, colorScale, x, y, display, layerClass, branchName = 'none') {
    const cellSize = 133 / image.length;
    const numRows = image.length;
    const numCols = image[0].length; 
    const strokeFill = (layerClass === 'ReLU') ? 'black' : (layerClass === '"BatchNorm2d"') ? 'black' : 'gray'
    // const strokeWidth = 1;
    const strokeWidth = (layerClass === 'ReLU') ? 3 : (layerClass === '"BatchNorm2d"') ? 3 : 1;
    const className = (branchName === 'none') ? `IntermediateResult-${layerClass}`:  `IntermediateResult-${branchName}-${layerClass}`
    let detailSVG = d3.select('#detail-svg');
    
    if(branchName !== 'none'){
      detailSVG = d3.select('#detail-svg').select(`g#${branchName}`);
    }    
    const imageCells = detailSVG.append('g')
    .attr('class', className)
    .attr('id', `IR-${visibleLayerIndex}-${layerIndex}-${imageIndex}`)
    .attr('transform', `translate(${x}, ${y})`)
    .style('display', display);

    
    image.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        const cell = imageCells.append('rect')
          .attr('x', colIndex * cellSize)
          .attr('y', rowIndex * cellSize)
          .attr('id', `blcok-${rowIndex}-${colIndex}`)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .style('fill', colorScale(value));
          
        // Adding Outline for Edge Cells
        if (rowIndex === 0) { // Top
          drawLine(colIndex, rowIndex, colIndex + 1, rowIndex, strokeFill, strokeWidth);
        }
        if (rowIndex === numRows - 1) { // Bottom
          drawLine(colIndex, rowIndex + 1, colIndex + 1, rowIndex + 1, strokeFill, strokeWidth);
        }
        if (colIndex === 0) { // Left
          drawLine(colIndex, rowIndex, colIndex, rowIndex + 1, strokeFill, strokeWidth);
        }
        if (colIndex === numCols - 1) { // Right
          drawLine(colIndex + 1, rowIndex, colIndex + 1, rowIndex + 1, strokeFill, strokeWidth);
        }
      });
    });
    function drawLine(x1, y1, x2, y2, strokeFill, strokeWidth) {
      imageCells.append('line')
        .attr('x1', x1 * cellSize)
        .attr('y1', y1 * cellSize)
        .attr('x2', x2 * cellSize)
        .attr('y2', y2 * cellSize)
        .style('stroke', strokeFill)
        .style('stroke-width', strokeWidth)
        .style('stroke-opacity', 1);
    }
  }


  function getLayerMaxMin3D(layerImages) {
      let flattenedData = layerImages.flat(Infinity);
      const max = flattenedData.reduce((a, b) => Math.max(a, b));
      const min = flattenedData.reduce((a, b) => Math.min(a, b));

      return [max,min];
  }
  function getLayerMaxMin(layerImages) {
      const flattenedData = layerImages.flat();
      const max = flattenedData.reduce((a, b) => Math.max(a, b));
      const min = flattenedData.reduce((a, b) => Math.min(a, b));

      return [max,min];
  }


  function toggleReLU(){
    let detailSVG = d3.select('#detail-svg');
    if(typeof selectedBranch !== 'undefined'){
      detailSVG = d3.select(`g#${selectedBranch}`);
    }
    detailSVG.selectAll('g.IntermediateResult-ReLU').each(function() {
        const id = d3.select(this).attr('id');
        const parts = id.split('-');
        const num = parseInt(parts[1], 10); 
        const convLayerId = `IR-${parts[1]}-0-${parts[3]}`;
        const convLayer = d3.select(`#${convLayerId}`);

        if(!reluActive){      
          convLayer.transition().duration(1000).style('display', 'none')
              .on('end', () => {
                  d3.select(this).transition().duration(1000).style('display', 'inline');
              });
        }
        else{
          d3.select(this).transition().duration(1000).style('display', 'none')
              .on('end', () => {
                convLayer.transition().duration(1000).style('display', 'inline');
              });
        }
    });
  }
  function toggleBN(){
    let detailSVG = d3.select('#detail-svg');
    let bnClassSelector = 'g.IntermediateResult-BatchNorm2d';
    if(typeof selectedBranch !== 'undefined'){
      detailSVG = d3.select(`g#${selectedBranch}`);
      bnClassSelector = `g.IntermediateResult-${selectedBranch}-BatchNorm2d`;
    }
    detailSVG.selectAll(bnClassSelector).each(function() {
        const id = d3.select(this).attr('id');
        const parts = id.split('-');
        const convLayerId = `IR-${parts[1]}-0-${parts[3]}`;
        const convLayer = d3.select(`#${convLayerId}`);

        if(!batchNormActive){      
          convLayer.transition().duration(1000).style('display', 'none')
              .on('end', () => {
                  d3.select(this).transition().duration(1000).style('display', 'inline');
              });
        }
        else{
          d3.select(this).transition().duration(1000).style('display', 'none')
              .on('end', () => {
                convLayer.transition().duration(1000).style('display', 'inline');
              });
        }
    });
  }
</script>

<style>
#svg-container {
    overflow: auto;
    height: 100%;
    width: 100%;
}
</style>

<Row>
  <Col sm="auto" class="d-flex align-items-center">
    <FormGroup class="d-flex align-items-center mb-0">
      <Label for="model-select" class="me-2 mb-0">Model</Label>
      <Input type="select" bind:value={selectedModel} id="model-select" class="me-3">
        {#each imagenetModels as modelName}
          <option value={modelName}>{modelName}</option>
        {/each}
      </Input>
    </FormGroup>
  </Col>

  <Col sm="auto" class="d-flex align-items-center">
    <FormGroup class="d-flex align-items-center mb-0">
      <Label for="class-select" class="me-2 mb-0">Class</Label>
      <Input type="select" bind:value={selectedClass} id="class-select" class="me-3">
        {#each Object.entries(imagenetClasses) as [index, className]}
          <option value={index}>{index}: {className}</option>
        {/each}
      </Input>
    </FormGroup>
  </Col>

  <Col sm="auto" class="d-flex align-items-center">
    <Button color="primary" on:click={loadModelView}>Load</Button>
  </Col>
</Row>

<div id="model-load"></div>

<Modal isOpen={openModal} toggle={closeDetailView} size='lg'>
  <ModalHeader toggle={closeDetailView}>
    <p>{selectedModel} detail view for {selectedClass} class - {selectedModule} module</p>
    {#if selectedModule == "inception"}
      <div class="d-flex">
        <Input type="select" bind:value={selectedBranch} id="branch-select" class="me-3" style="width: auto;">
          {#each branches as branch}
            <option value={branch}>{branch}</option>
          {/each}
        </Input>
      </div>
    {/if}
  </ModalHeader>
  <ModalBody>
    <div id="svg-container">
      <svg id="detail-svg"></svg>
    </div>
  </ModalBody>
  <ModalFooter class="d-flex justify-content-end">
    <div class="switch-container d-flex align-items-center">
      <FormCheck type="switch" id="form-ReLU" label="ReLU" bind:checked={reluActive} on:change={toggleReLU} />
      <FormCheck type="switch" id="form-BN" label="BatchNorm" bind:checked={batchNormActive} on:change={toggleBN} />
    </div>
  </ModalFooter>
</Modal>

<!-- <div id='overview'>
	<Overview />
</div> -->