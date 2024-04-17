<script>
	import { onMount, tick } from 'svelte';
  import * as d3 from 'd3';
  import { Container, Input, FormGroup, Label, FormCheck, Button, Row, Col, Modal, ModalBody, ModalHeader, ModalFooter } from 'sveltestrap';
  import Header from './Header.svelte'

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
    modelSVG = d3.select('#model-container').select('svg');
    link = d3.linkHorizontal()
      .x(d=>d[0])
      .y(d=>d[1]);
  });

  let modelData = undefined;
  const moduleXMargin = 20;
  const moduleYMargin = 200;
  const moduleWidth = 100;
  const moduleHeight = 400;

  const layerWidth = moduleWidth * 0.8
  const layerHeight = moduleHeight * 0.8
  const layerXOffset = (moduleWidth - layerWidth) / 2
  const layerYOffset = (moduleHeight - layerHeight) / 2

  let hoveredSoftmaxBlock = undefined;
  // let tooltipVisible = false;
  let tooltipX = 0;
  let tooltipY = 0;
  let top5Index = undefined;
  let softmaxProbs = undefined;
  let isHuggingFaceModel = false;
  let isUserInputImage = false;
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

    let moduleStruct = [];
    for (const key in modelData) {
      let layerInfo = modelData[key]
      let moduleIdx = layerInfo['module_index']
      let moduleName = layerInfo['module_name']
      let layerName = layerInfo['class']

      if (moduleStruct[moduleIdx] === undefined){ //
        moduleStruct.push({
          'name': moduleName,
          'layers': []
        });
      }

      moduleStruct[moduleIdx]['layers'].push({
        'name': layerName
      }); 
    }

    const moduleGroup = modelSVG.append('g').attr('class', 'module-group');

    const modules = moduleGroup.selectAll('g')
      .data(moduleStruct)
      .enter()
      .append('g')
      .attr('class', 'module')
      .attr('transform', (d, i) => `translate(${i * (moduleWidth + moduleXMargin)}, ${moduleYMargin})`);

    // 각 하위 g 요소 안에 rect 추가
    modules.append('rect')
      .attr('width', moduleWidth)
      .attr('height', moduleHeight)
      .style('fill', (d) => moduleFills(d['name']))
      .style('stroke', 'gray')
      .style('stroke-width', 0);

    // 각 하위 g 요소 안에 text 추가
    modules.append('text')
      .attr('x', moduleWidth / 2)
      .attr('y', moduleHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .text((d) => d['name'])
      .style('fill', 'black');

      modules.each(function(d, i) {
        const group = d3.select(this);

        group.select('rect')
          .on('mouseover', function() {
            const expandedWidth = moduleWidth * d['layers'].length; // 확장할 너비
            const shiftDistance = expandedWidth - moduleWidth; // 확장으로 인해 밀어낼 거리

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
              .attr('transform', (d, j) => `translate(${((j + i + 1) * (moduleWidth + moduleXMargin)) + shiftDistance}, ${moduleYMargin})`);

            // Hide text
            group.select('text')
              .transition()
              .duration(500)
              .style('opacity', 0);

            // Create empty group within rect
            group.append('g')
              .attr('class', 'layer-group')

            // Add layers inside layer group
            const layers = group.select('g.layer-group').selectAll('g')
              .data(d['layers'])
              .enter()
              .append('g')
              .attr('class', 'layer')
              .attr('transform', (d, i) => `translate(${i * (layerWidth + moduleXMargin) + layerXOffset}, ${layerYOffset})`)
              .style('pointer-events','none')

            layers.append('rect')
              .attr('width', layerWidth)
              .attr('height', layerHeight)
              .style('fill', 'white')
              .style('stroke', 'gray')
              .style('stroke-width', 0);

            layers.append('text')
              .attr('x', layerWidth / 2)
              .attr('y', layerHeight / 2)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .text((d) => d['name'])
              .style('fill', 'black');
            })
          .on('mouseout', function() {
            // 모든 rect를 원래 크기로 복원
            d3.select(this)
              .transition()
              .duration(500)
              .attr('width', moduleWidth)
              .style('stroke-width', 1);
            
            d3.select('g.layer-group').remove();

            // 모든 rect를 원래 위치로 복원
            modules.transition()
              .duration(500)
              .attr('transform', (d, j) => `translate(${j * (moduleWidth + moduleXMargin)}, ${moduleYMargin})`);

            // 모든 text를 다시 표시
            modules.select('text')
              .transition()
              .duration(500)
              .style('display', 'inline')
              .style('opacity', 1);
        })
        //Click Effect => Call 'showDetailView'
        .on('click', () => { 
          openDetailView(d['name'], i); 
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
    // modelStruct
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
      drawLayerConnections();
      setLayerEvents();
    }
    else if (moduleName === 'avgpool'){
      drawAvgpoolModuleDetail(moduleLayers);
      drawLayerConnections();
      setLayerEvents();
    }
    else if (moduleName === 'linear'){
      drawLinearModuleDetail(moduleLayers, inputLayer);
      setLinearLayerEvents();
    }
    else if (moduleName === 'residual'){
      drawResidualModuleDetail(moduleLayers, layerNames);
      drawLayerConnections();
      drawShortCuts();
      setLayerEvents();
    }
    else if (moduleName === 'inception'){
      drawInceptionModuleDetail(moduleLayers, layerNames);
      drawLayerConnections();
      setLayerEvents();
    }
  }

  function drawShortCuts(){
    const residualLayer = d3.select('#detail-svg').selectAll('g.IntermediateResult-Residual');
    const stepBeforeLine = d3.line()
                            .x(d => d[0])
                            .y(d => d[1])
                            .curve(d3.curveStepBefore);
    const stepAfterLine = d3.line()
                            .x(d => d[0])
                            .y(d => d[1])
                            .curve(d3.curveStepAfter);
    residualLayer.each(function() {
      const dstIR = d3.select(this);
      const idTokens = dstIR.attr('id').split('-');
      const dstLayerIndex = idTokens[1];
      const dstImageIndex = idTokens[3];
      const srcIR = d3.select('#detail-svg').select(`g#IR-0-0-${dstImageIndex}`);

      const srcGroupTranslate = srcIR.attr('transform').match(/translate\(([^)]+)\)/);
      const srcCoordinates = srcGroupTranslate[1].split(',').map(function(d) { return parseFloat(d); });

      const dstGroupTranslate = dstIR.attr('transform').match(/translate\(([^)]+)\)/);
      const dstCoordinates = dstGroupTranslate[1].split(',').map(function(d) { return parseFloat(d); });

      const leftX = srcCoordinates[0] + imageWidth / 2;
      const rightX = dstCoordinates[0] + imageWidth / 2;
      const bottomY = srcCoordinates[1];
      const topY = bottomY - offsetY / 2;

      const pathDataBeforeLine = [
        [leftX, bottomY],
        [(leftX + rightX) / 2, topY]
      ];

      const pathDataAfterLine = [
        [(leftX + rightX) / 2, topY],
        [rightX, bottomY]
      ];
        
      d3.select('#detail-svg').append('path')
        .attr('d', stepBeforeLine(pathDataBeforeLine))
        .attr('fill', 'none')
        .attr('stroke', 'gray')
        .attr('class','residual-edge')
        .attr('id', `edge-${dstLayerIndex}-${dstImageIndex}-${dstImageIndex}`)
        .attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);
        
        d3.select('#detail-svg').append('path')
        .attr('d', stepAfterLine(pathDataAfterLine))
        .attr('fill', 'none')
        .attr('stroke', 'gray')
        .attr('class','residual-edge')
        .attr('id', `edge-${dstLayerIndex}-${dstImageIndex}-${dstImageIndex}`)
        .attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);
    });
  }

  function drawLayerConnections(){
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
            addImageConnection(srcIR, dstIR, cursor, srcImageIndex ,dstImageIndex);
          });
        }
        else if(layerClass === 'IntermediateResult-Concat'|| layerClass === 'IntermediateResult-Residual' || layerClass.includes('Pool')){  //ToDo(YSKIM): residual -> Add 
          const srcIR = prevLayer.filter(function() {
            return d3.select(this).attr('id').split('-')[3] === dstImageIndex;
          })
          addImageConnection(srcIR, dstIR, cursor, dstImageIndex ,dstImageIndex);
        }
      });
    }
    pathData.forEach(path => {
      d3.select('#detail-svg').append('path')
          .attr('d', link({
            source: path.source,
            target: path.target
          }))
          .attr('class','edge')
          .attr('fill', 'none')
          .attr('stroke', 'gray')
          .attr('stroke-width', 1)
          .attr('id', path.id)
          .style('stroke-opacity', 0.5);
    });
  }


  function setLayerEvents(){
    const IRs = d3.select('#detail-svg').selectAll('g').filter(function() { return this.getAttribute('class').includes('IntermediateResult') });
    let paths = undefined;
    
    IRs.on('mouseover', function() {
      const idTokens = d3.select(this).attr('id').split('-');
      const layerIndex = idTokens[1];
      const IRIndex = idTokens[3];
      const IRClass = d3.select(this).attr('class');

      // const paths = d3.select('#detail-svg').selectAll('path').filter(function() {
      paths = d3.select('#detail-svg').selectAll('path').filter(function() {
        const edgeClass = d3.select(this).attr('class');
        const edgeIndex = d3.select(this).attr('id').split('-');
        const isNextLayerPath = (parseInt(edgeIndex[1]) - 1 === parseInt(layerIndex)) && (edgeIndex[2] === IRIndex);
        const isPrevLayerPath = (edgeIndex[1] === layerIndex) && (edgeIndex[3] === IRIndex);
        const isNextLayerNotResidual = (parseInt(edgeIndex[1]) - 1 === parseInt(layerIndex)) && (edgeClass === 'edge');

        return (isNextLayerPath && isNextLayerNotResidual) || isPrevLayerPath;
      });

      paths.attr('stroke-width', 3)
        .style('stroke-opacity', 0.7);

      
      const strokeFill = (IRClass.includes('ReLU')) ? 'black' : (IRClass.includes('BatchNorm2d')) ? 'black' : 'gray'

      d3.select(this).append('rect')
        .attr('class', 'IR-wrapper')
        .attr('width', imageWidth)
        .attr('height', imageHeight)
        .attr('fill', 'none')
        .attr('stroke', strokeFill)
        .attr('stroke-width', 3)
        .style('stroke-opacity', 1);

    }).on('mouseout', function() {
      const layerIndex = d3.select(this).attr('id').split('-')[1];
      const IRIndex = d3.select(this).attr('id').split('-')[3];

      paths.attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);

      d3.select(this).select('rect.IR-wrapper').remove();
    });
  }

  function setLinearLayerEvents(){
    const softmaxBlocks = d3.select('#detail-svg').select('g.Intermediate-Softmax').selectAll('rect.block');
    const blocks = d3.select('#detail-svg').selectAll('rect.block');
    
    // d3.select('#detail-svg').on('click', function() {
    //   d3.select(this).selectAll('path').remove();
    //   // tooltipLock = true;
    //   // tooltipVisible = false;
    // })

    //softmax block event handling    
    softmaxBlocks.on('mouseover', function() {
      const hoveredLabelIndex = d3.select(this).attr('id').split('-')[1];
      // hoveredSoftmaxLabel = {class:prob}
      // tooltipVisible = true;
    }).on('mouseout', function() {
      hoveredSoftmaxBlock = undefined;
      // tooltipVisible = false;
    });

    //linear block event handling
    blocks.on('mouseover', function() {
      d3.select(this).style('stroke-width', 3);
    }).on('mouseout',function() {
      d3.select(this).style('stroke-width', 1);
    }).on('click', function() {
      d3.select('#detail-svg').selectAll('path').remove();
      pathData = [];

      const selectedBlock = d3.select(this);
      const selectedLayerDepth = parseInt(d3.select(this.parentNode).attr('id').split('-')[1]);
      const selectedBlockIndex = selectedBlock.attr('id').split('-')[1];

      //select PrevLayer Rects
      if(selectedLayerDepth > 0){
        const prevBlocks = d3.select('#detail-svg').selectAll('g').filter(function(){
          if(!this.getAttribute('class').includes('IntermediateResult')){
            return false;
          }
          const isPrevLayer = parseInt(d3.select(this).attr('id').split('-')[1]) === (selectedLayerDepth - 1);
          const isDisplayInline = window.getComputedStyle(this).display === 'inline';

          return isPrevLayer && isDisplayInline;
        }).selectAll('rect');

        prevBlocks.each(function() {
            const prevBlock = d3.select(this);
            const prevBlockIndex = prevBlock.attr('id').split('-')[1];
            addBlockConnection(prevBlock, selectedBlock, selectedLayerDepth, prevBlockIndex, selectedBlockIndex);
        });
      }
      //select NextLayer Rects
      if(selectedLayerDepth < moduleLayerDepth){
        const nextBlocks = d3.select('#detail-svg').selectAll('g').filter(function(){
          if(!this.getAttribute('class').includes('IntermediateResult')){
            return false;
          }
          const isNextLayer = parseInt(d3.select(this).attr('id').split('-')[1]) === (selectedLayerDepth + 1);
          const isDisplayInline = window.getComputedStyle(this).display === 'inline';

          return isNextLayer && isDisplayInline;
        }).selectAll('rect');
        
        nextBlocks.each(function() {
            const nextBlock = d3.select(this);
            const nextBlockIndex = nextBlock.attr('id').split('-')[1];
            addBlockConnection(selectedBlock, nextBlock, selectedLayerDepth + 1, selectedBlockIndex, nextBlockIndex);
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
          .style('stroke-opacity', 0.3);
      });
    });
  }
  function addBlockConnection(srcBlock, dstBlock, dstLayerIndex, srcBlockIndex, dstBlockIndex){
    const srcGroupTranslate = srcBlock.node().parentNode.getAttribute('transform').match(/translate\(([^)]+)\)/);
    let srcCoordinates = srcGroupTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    srcCoordinates[0] = srcCoordinates[0] + parseFloat(srcBlock.attr('x')) + parseFloat(srcBlock.attr('width'));
    srcCoordinates[1] = srcCoordinates[1] + parseFloat(srcBlock.attr('y')) + parseFloat(srcBlock.attr('height')) / 2;

    const dstGroupTranslate = dstBlock.node().parentNode.getAttribute('transform').match(/translate\(([^)]+)\)/);
    let dstCoordinates = dstGroupTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    dstCoordinates[0] = dstCoordinates[0] + parseFloat(dstBlock.attr('x'));
    dstCoordinates[1] = dstCoordinates[1] + parseFloat(dstBlock.attr('y')) + parseFloat(dstBlock.attr('height')) / 2;

    pathData.push({ id:`edge-${dstLayerIndex}-${srcBlockIndex}-${dstBlockIndex}`, source: srcCoordinates, target: dstCoordinates}) 
  }

  function addImageConnection(srcImage, dstImage, dstLayerIndex, srcImageIndex, dstImageIndex){
    const srcTranslate = srcImage.attr('transform').match(/translate\(([^)]+)\)/);
    let srcCoordinates = srcTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    srcCoordinates[0] = srcCoordinates[0] + imageWidth;
    srcCoordinates[1] = srcCoordinates[1] + imageHeight / 2;

    const dstTranslate = dstImage.attr('transform').match(/translate\(([^)]+)\)/);
    let dstCoordinates = dstTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    dstCoordinates[1] = dstCoordinates[1] + imageHeight / 2;

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
        // const inputX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        // const inputY = moduleYPadding + (offsetY);
        // drawLayer(inputLayer, visibleLayerIndex, hiddenLayerCount, inputX, inputY, 'inline', 'input');

        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        // drawFlatten3D(inputLayer, 0, 0, x, y, 'inline', 'input');
        drawLinear(layer['input'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class']);
      }
      //Last Layer contains Top-10 prediction labes (output_index) and probability (output)
      if(layerIndex === (moduleLayers.length - 1)){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        //TODO(YSKIM): Print top 10 labels
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        
        softmaxProbs = layer['softmax_output'];
        top5Index = layer['output_index'];

        drawLinear(layer['softmax_output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', 'Softmax');
      }
      //ReLU Layer
      else if(layer['class'] === 'ReLU'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLinear(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['class']);
      }
      //Other Layers (Linear)
      else if(layer['class'] === 'Linear'){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
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
    console.log(layerNames)
    console.log(moduleLayers)
    moduleLayers.forEach((layer, layerIndex) => {
      //last RelU Layer includes identity
      if(layerIndex === (moduleLayers.length - 1)){   
        const inputX = moduleXPadding;
        const inputY = moduleYPadding + (offsetY);
        drawLayer(layer['identity'], 0, 0, inputX, inputY, 'inline', 'input');

        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        // const x1 = inputX + imageWidth/2;
        // const x2 = x + imageWidth/2;
        // drawShortcut(x1, x2, y, 1);
        drawLayer(layer['input'], visibleLayerIndex, 0, x, y, 'inline', 'Residual');
        drawLayer(layer['output'], visibleLayerIndex, 1, x, y, 'none', layer['class']);
      }
      else if(layerNames[layerIndex].includes('downsample')){
        //ToDo: Add downsampling IR result
      } 
      else if(layer['class'] === 'ReLU' || layer['class'] === 'BatchNorm2d'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);

        visibleLayerIndex = visibleLayerIndex;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['class']);
      }
      else if(layer['class'] === 'Conv2d' || layer['class'].includes('Pool')){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['class']);
      }
    });
    moduleLayerDepth = visibleLayerIndex;
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
        .attr('class','block')
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
    const strokeFill = (layerClass === 'ReLU') ? 'black' : (layerClass === 'BatchNorm2d') ? 'black' : 'gray'
    // const strokeWidth = 1;
    const strokeWidth = 1;
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
        const tokens = id.split('-');
        const num = parseInt(tokens[1], 10); 
        const LayerId = `IR-${tokens[1]}-0-${tokens[3]}`;
        const Layer = d3.select(`#${LayerId}`);

        if(!reluActive){      
          Layer.transition().style('display', 'none')
              .on('start', () => {
                  d3.select(this).transition().style('display', 'inline');
              });
        }
        else{
          d3.select(this).transition().style('display', 'none')
              .on('start', () => {
                Layer.transition().style('display', 'inline');
              });
        }
    });
  }
  function toggleBN(){
    let detailSVG = d3.select('#detail-svg');
    let bnClassSelector = 'g.IntermediateResult-BatchNorm2d';
    if(typeof selectedBranch !== 'undefined'){
      detailSVG = d3.select('#detail-svg').select(`g#${selectedBranch}`);
      bnClassSelector = `g.IntermediateResult-${selectedBranch}-BatchNorm2d`;
    }
    detailSVG.selectAll(bnClassSelector).each(function() {
        const id = d3.select(this).attr('id');
        const tokens = id.split('-');
        const LayerId = `IR-${tokens[1]}-0-${tokens[3]}`;
        const Layer = detailSVG.select(`#${LayerId}`);

        if(!batchNormActive){      
          Layer.transition().style('display', 'none')
              .on('start', () => {
                  d3.select(this).transition().style('display', 'inline');
              });
        }
        else{
          d3.select(this).transition().style('display', 'none')
              .on('start', () => {
                Layer.transition().style('display', 'inline');
              });
        }
    });
  }

function handleFileChange() {


}
</script>

<style>
  #svg-container {
      overflow: auto;
      height: 100%;
      width: 100%;
  }
</style>

<Header/>

<Container fluid>
  <Row class="h-100">
    <Col class="d-flex flex-column" style="flex: 0 0 400px; max-width: 400px; height: 100vh; overflow-y: auto; background-color: rgba(249,249,249,255);">
      <Row class="d-flex align-items-center" style="padding: 2.5px 10px;">
        <FormCheck type="switch" id="form-model" label="Hugging Face Model URL" bind:checked={isHuggingFaceModel} />
      </Row>
      <Row class="d-flex align-items-center" style="padding: 2.5px 10px;">
        <FormGroup class="d-flex align-items-center mb-0">
          <Label for="model-select" class="me-2 mb-0">Model</Label>
          <Input type="select" bind:value={selectedModel} id="model-select" class="me-3" disabled={isHuggingFaceModel}>
            {#each imagenetModels as modelName}
              <option value={modelName}>{modelName}</option>
            {/each}
          </Input>
        </FormGroup>
      </Row>
      <Row class="d-flex align-items-center" style="padding: 2.5px 10px;">
        <FormGroup class="d-flex align-items-center mb-0">
          <Label for="HuggingFace-url" class="me-2 mb-0">URL</Label>
          <Input type="text" id="model-url" placeholder="Type 'microsoft/resnet-18' here" class="me-3" disabled={!isHuggingFaceModel} />
        </FormGroup>
      </Row>
      <Row class="d-flex align-items-center" style="padding: 2.5px 10px;">
        <FormCheck type="switch" id="form-model" label="User Image Input" bind:checked={isUserInputImage} />
      </Row>
      <Row class="d-flex align-items-center" style="padding: 2.5px 10px;">
        <FormGroup class="d-flex align-items-center mb-0">
          <Label for="class-select" class="me-2 mb-0">Class</Label>
          <Input type="select" bind:value={selectedClass} id="class-select" class="me-3" disabled={isUserInputImage}>
            {#each Object.entries(imagenetClasses) as [index, className]}
              <option value={index}>{index}: {className}</option>
            {/each}
          </Input>
        </FormGroup>
      </Row>

      <Row class="d-flex align-items-center">
        <FormGroup class="d-flex align-items-center mb-0">
          <Label for="class-select" class="me-2 mb-0">Image Input</Label>
          <Input type="file" id="image-upload" accept="image/*" on:change={handleFileChange} disabled={!isUserInputImage} />
        </FormGroup>
      </Row>
    
      <Row class="d-flex align-items-center">
        <Button color="secondary" on:click={loadModelView}>Load</Button>
      </Row>

      <Row class="mt-auto">
      </Row>


    </Col>
    <Col style="flex-grow: 1; height: 100vh;">
      <div id="model-container" style="width: 100%; height: 100%; overflow: auto;">
        <svg style="width: 100%; height: 100%; display: block; min-width: 100%; min-height: 100%;">
        </svg>
      </div>
    </Col>
  </Row>
</Container>

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
      <svg id="detail-svg">
      </svg>
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