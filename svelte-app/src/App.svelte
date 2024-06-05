<script>
	import { onMount, tick } from 'svelte';
  import * as d3 from 'd3';
  import { interpolateRdBu } from 'd3-scale-chromatic';
  import { Container, Input, FormGroup, Label, FormCheck, Button, Row, Col, Modal, ModalBody, ModalHeader, ModalFooter } from 'sveltestrap';
  import Header from './Header.svelte'
    // import { construct_svelte_component } from 'svelte/internal';
  // import 'bootstrap/dist/css/bootstrap.min.css';

    // import { construct_svelte_component } from 'svelte/internal';

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
  let stepBeforeLine = undefined;
  let stepAfterLine = undefined;
  let straightLine = undefined;

  let branches = [];
  const imagenetModels = ['alexnet', 'vgg16', 'googlenet', 'resnet18'];
  let imagenetClasses ={};

  onMount(async () => {
    const response = await fetch('/imageClasses.json');
    imagenetClasses = await response.json();
    selectedModel = imagenetModels[0];
    selectedClass = "0";
    modelSVG = d3.select('#model-container').select('svg');
    const modelZoom = d3.zoom()
      .scaleExtent([0.5, 2])  // zoom range
      .on('zoom', (event) => {
        modelSVG.select('g#model-structure').attr('transform', event.transform);
      });

    modelSVG.call(modelZoom);

    const moduleSVG = d3.select('#module-container').select('svg');
    const moduleZoom = d3.zoom()
      .scaleExtent([0.5, 2])  // zoom range
      .on('zoom', (event) => {
        moduleSVG.select('g#module-structure').attr('transform', event.transform);
      });

    moduleSVG.call(moduleZoom);

    link = d3.linkHorizontal()
    .x(d=>d[0])
    .y(d=>d[1]);

    stepBeforeLine = d3.line()
                    .x(d => d[0])
                    .y(d => d[1])
                    .curve(d3.curveStepBefore);
    stepAfterLine = d3.line()
                    .x(d => d[0])
                    .y(d => d[1])
                    .curve(d3.curveStepAfter);
    straightLine = d3.line()
                    .x(d => d[0])
                    .y(d => d[1]);
  });

  let modelData = undefined;
  const moduleXMargin = 20;
  const moduleYMargin = 200;
  const moduleWidth = 100;
  const moduleHeight = 400;

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
    d3.select('#module-svg')
      .attr('width', newWidth + 500)
      .attr('height', newHeight);
  }

  async function loadModelView() {
    modelSVG.select('g#model-structure').selectAll('*').remove();
    const response = await fetch(`/output/${selectedClass}/${selectedModel}_info.json`);
    modelData = await response.json();
    console.log("Loaded JSON data:", modelData);

    // JSON 객체의 모든 키를 출력
    console.log("Keys in JSON:", Object.keys(modelData));

    const moduleGroup = modelSVG.select('g#model-structure').append('g').attr('class', 'module-group');
    const modules = moduleGroup.selectAll('g')
      .data(modelData)
      .enter()
      .append('g')
      .attr('class', 'module')
      .attr('transform', (d, i) => `translate(${i * (moduleWidth + moduleXMargin)}, ${moduleYMargin})`)
      .style('cursor', 'pointer')
      .on('mousedown', function(event) {
        event.stopPropagation();
      });;

    // 각 하위 g 요소 안에 rect 추가
    modules.append('rect')
      .attr('width', moduleWidth)
      .attr('height', moduleHeight)
      .style('fill', (d) => moduleFills(d['type']))
      .style('stroke', 'gray')
      .style('stroke-width', 0);

    // 각 하위 g 요소 안에 text 추가
    modules.append('text')
      .attr('x', moduleWidth / 2)
      .attr('y', moduleHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .text((d) => d['type'])
      .style('fill', 'black');

      modules.each(function(d, i) {
        const group = d3.select(this);

        group.select('rect')
          .on('mouseover', function() {
            let maxLength = 0
            const layerLength = d['layers'].length;
            const numBranch = d['branches'].length;
            
            for (let i = 0; i < numBranch; i++) {
              maxLength = Math.max(maxLength, d['branches'][i].length)
            }
            // console.log(`Max layers in this module ${maxLength}`)
            const expandedWidth = moduleWidth * (maxLength + layerLength); // 확장할 너비
            const shiftDistance = expandedWidth - moduleWidth; // 확장으로 인해 밀어낼 거리
            const layerWidth = moduleWidth * 0.8;
            const layerHeight = numBranch == 1 ? moduleHeight * 0.8 : moduleHeight * 0.7 / numBranch;
            const layerXOffset = (moduleWidth - layerWidth) / 2;
            const layerYOffset = (moduleHeight - layerHeight) / 2;
            const layerYMargin = numBranch == 1 ? moduleHeight * 0.2 / 2 : moduleHeight * 0.3 / (numBranch + 1);

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
              .attr('class', 'branch-group')
            
            // Add layers inside branch group
            const branch = group.select('g.branch-group').selectAll('g')
              .data(d['branches'])
              .enter()
              .append('g')
              .attr('class', 'branch')
              .attr('transform', (_, i) => `translate(${0}, ${i * (layerHeight + layerYMargin) + layerYMargin})`)
              .style('pointer-events','none');

            branch.append('g')
              .attr('class', 'layer-group');

            const branchLayers = branch.select('g.layer-group').selectAll('g')
              .data((d) => d)
              .enter()
              .append('g')
              .attr('class', 'branch-layer')
              .attr('transform', (_, i) => `translate(${i * (layerWidth + moduleXMargin) + layerXOffset}, ${0})`)

            //Branch layers
            branchLayers.append('rect')
              .transition()
              .delay(500)
              .attr('width', layerWidth)
              .attr('height', layerHeight)
              .attr('id', (_, j) => `branch-${i}-${j}`)
              .style('fill', 'white')
              .style('stroke', 'gray')
              .style('stroke-width', 0)
              .attr('fill-opacity', 1);

            branchLayers.append('text')
              .transition()
              .delay(500)
              .attr('x', layerWidth / 2)
              .attr('y', layerHeight / 2)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .text((d) => d['layer_type'])
              .style('fill', 'black')
              .attr('fill-opacity', 1);

            //Remain layers
            const layers = group.select('g.layer-group').selectAll('g.layer')
              .data((d) => d['layers'])
              .enter()
              .append('g')
              .attr('class','layer')
              .attr('id', (_, j) => `layer-${j}`)
              .attr('transform', (_, i) => `translate(${(i + maxLength) * (layerWidth + moduleXMargin) + layerXOffset}, ${0})`);

            layers.append('rect')
              .transition()
              .delay(500)
              .attr('width', layerWidth)
              .attr('height', () => numBranch == 1 ? layerHeight : numBranch * (layerHeight + layerYMargin) - layerYMargin )
              .style('fill', 'white')
              .style('stroke', 'gray')
              .style('stroke-width', 0)
              .attr('fill-opacity', 1);
             
            const parentGroup = d3.selectAll('g.module').nodes()[i];
            for(let j = 0; j < d['branches'].length; j++){
              for(let k = 0; k <= d['branches'][j].length; k++){
                const srcX = k == 0 ? 0 : k * (layerWidth + moduleXMargin) - layerXOffset;
                const initY = moduleYMargin + d3.select(parentGroup).select('rect').attr('hegiht') / 2;
                const dstX = k * (layerWidth + moduleXMargin) + layerXOffset;
                const dstY = j * (layerHeight + layerYMargin) + layerYMargin + layerHeight / 2;

                // start edge
                if(k == 0){
                  const initPath = [
                    [srcX, initY], [dstX / 2, dstY], [dstX, dstY]
                  ];                  
                  d3.select(parentGroup).append('path')
                  .attr('d',stepAfterLine(initPath))
                  .attr('fill', 'none')
                  .transition()
                  .delay(500)
                  .attr('stroke', 'black')
                  .attr('class','module-edge')
                  .attr('stroke-width', 1)
                  .style('stroke-opacity', 1);
                }
                // last edge of branches
                if(d['branches'][j].length == 0 || k == d['branches'][j].length){
                  const modulePathData = []
                  modulePathData.push([dstX, dstY]);
                  modulePathData.push([maxLength * (layerWidth + moduleXMargin) + layerXOffset, dstY]);

                  d3.select(parentGroup).append('path')
                  .attr('d',straightLine(modulePathData))
                  .attr('fill', 'none')
                  .transition()
                  .delay(500)
                  .attr('stroke', 'black')
                  .attr('class','module-edge')
                  .attr('stroke-width', 1)
                  .style('stroke-opacity', 1);
                }
                // branch edges
                if(k > 0){
                  const modulePathData = []
                  modulePathData.push([srcX, dstY]);
                  modulePathData.push([dstX, dstY]);

                  d3.select(parentGroup).append('path')
                  .attr('d',straightLine(modulePathData))
                  .attr('fill', 'none')
                  .transition()
                  .delay(500)
                  .attr('stroke', 'black')
                  .attr('class','module-edge')
                  .attr('stroke-width', 1)
                  .style('stroke-opacity', 1);
                }
              }
            }
            for(let j = 0; j < d['layers'].length; j++){
              const srcX = maxLength * (layerWidth + moduleXMargin) + (j + 1) * (layerWidth + moduleXMargin) - layerXOffset;
              const initY = moduleYMargin + d3.select(parentGroup).select('rect').attr('hegiht') / 2;
              const dstX = j == d['layers'].length - 1 ? srcX + layerXOffset : srcX + moduleXMargin;

              const modulePathData = []
              modulePathData.push([srcX, initY]);
              modulePathData.push([dstX, initY]);

              d3.select(parentGroup).append('path')
                  .attr('d',straightLine(modulePathData))
                  .attr('fill', 'none')
                  .transition()
                  .delay(500)
                  .attr('stroke', 'black')
                  .attr('class','module-edge')
                  .attr('stroke-width', 1)
                  .style('stroke-opacity', 1);
            }
    
            layers.append('text')
              .transition()
              .delay(500)
              .attr('x', layerWidth / 2)
              .attr('y', moduleHeight * 0.8 / 2)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .text((d) => d['layer_type'])
              .style('fill', 'black')
              .attr('fill-opacity', 1);
            })
          .on('mouseout', function() {
            // 모든 rect를 원래 크기로 복원
            d3.select(this)
              .transition()
              .duration(500)
              .attr('width', moduleWidth)
              .style('stroke-width', 1);

            d3.select('svg#model-svg').selectAll('path').remove();
            d3.select('g.branch-group').remove();

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
          openDetailView(d, i); 
        });
      });


    function moduleFills(moduleName){
      if (moduleName === 'conv'){ return 'green'; }
      else if (moduleName === 'residual'){ return 'red'; }
      else if (moduleName === 'avgpool'){ return 'yellow';}
      else if (moduleName === 'linear'){ return 'orange'; }
      else if (moduleName === 'inception'){ return 'gray' }
    }
  }

  // Open Detail View of Selected Module
  async function openDetailView(selectedModuleInfo, selectedModuleIndex) {
    if(openModal){
      clearDetailView();
    }
    selectedModule = selectedModuleInfo['type'];
    openModal = true;
    await tick();

    drawModuleDetail(selectedModuleInfo, selectedModuleIndex);
  }
  // Close Detail View 
  function clearDetailView() {
    openModal = false;  
    reluActive = false;
    batchNormActive = false;
    selectedBranch = undefined;
    pathData = [];
    branches = [];
    d3.select('g#module-structure').selectAll("*").remove();
  }

  // Call drawModule functions depending on the type of module
  function drawModuleDetail(selectedModuleInfo, selectedModuleIndex){
    if (selectedModuleInfo['type'] === 'conv'){
      drawConvModuleDetail(selectedModuleInfo['layers']);
      drawLayerConnections();
      setLayerEvents();
    }
    else if (selectedModuleInfo['type'] === 'avgpool'){
      drawAvgpoolModuleDetail(selectedModuleInfo['layers']);
      drawLayerConnections();
      setLayerEvents();
    }
    else if (selectedModuleInfo['type'] === 'linear'){
      drawLinearModuleDetail(selectedModuleInfo['layers']);
      setLinearLayerEvents(selectedModuleInfo['layers']);
    }
    else if (selectedModuleInfo['type'] === 'residual'){
      drawResidualModuleDetail(selectedModuleInfo);
      drawLayerConnections();
      drawShortCuts();
      setLayerEvents();
    }
    else if (selectedModuleInfo['type'] === 'inception'){
      drawInceptionModuleDetail(selectedModuleInfo);
      setLayerEvents();
    }
  }

  function drawShortCuts(){
    const residualLayer = d3.select('#module-structure').selectAll('g.IntermediateResult-add');

    residualLayer.each(function() {
      const dstIR = d3.select(this);
      const idTokens = dstIR.attr('id').split('-');
      const dstLayerIndex = idTokens[1];
      const dstImageIndex = idTokens[3];
      const srcIR = d3.select('#module-structure').select(`g#IR-0-0-${dstImageIndex}`);

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
        
      d3.select('#module-structure').append('path')
        .attr('d', stepBeforeLine(pathDataBeforeLine))
        .attr('fill', 'none')
        .attr('stroke', 'gray')
        .attr('class','residual-edge')
        .attr('id', `edge-0-${dstImageIndex}-${dstLayerIndex}-${dstImageIndex}`)
        .attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);
        
        d3.select('#module-structure').append('path')
        .attr('d', stepAfterLine(pathDataAfterLine))
        .attr('fill', 'none')
        .attr('stroke', 'gray')
        .attr('class','residual-edge')
        .attr('id', `edge-0-${dstImageIndex}-${dstLayerIndex}-${dstImageIndex}`)
        .attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);
    });
  }

  function drawLayerConnections(){
    for(let cursor = moduleLayerDepth; cursor > 0; cursor--){
      const currLayer = d3.select('#module-structure').selectAll('g').filter(function() { 
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

      const prevLayer = d3.select('#module-structure').selectAll('g').filter(function() { 
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

        if (layerClass.includes('IntermediateResult') && layerClass.includes('conv')) {
          prevLayer.each(function() {
            const srcIR = d3.select(this);
            const srcImageIndex = srcIR.attr('id').split('-')[3];
            addImageConnection(srcIR, dstIR, cursor - 1, cursor, srcImageIndex ,dstImageIndex);
          });
        }
        else if(layerClass === 'IntermediateResult-add' || layerClass.includes('pool')){
          const srcIR = prevLayer.filter(function() {
            return d3.select(this).attr('id').split('-')[3] === dstImageIndex;
          })
          addImageConnection(srcIR, dstIR, cursor - 1, cursor, dstImageIndex ,dstImageIndex);
        }
      });
    }
    pathData.forEach(path => {
      d3.select('#module-structure').append('path')
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

  function drawInceptionLayerConnections(){
    const branchLayers = d3.select('#module-structure').select(`g#${selectedBranch}`).selectAll('g').filter(function(){
      const idTokens = d3.select(this).attr('id').split('-');
      return idTokens[2] == '0' && idTokens[3] == '0';
    });
    const layers = d3.select('#module-structure').selectAll('g').filter(function(){
      const idTokens = d3.select(this).attr('id').split('-');
      const isBranchLayer = d3.select(this).attr('class').includes('branch');
      return idTokens[2] == '0' && idTokens[3] == '0' && !isBranchLayer;
    });
    let layerOrder = [];
    branchLayers.each(function() {
      layerOrder.push(parseInt(d3.select(this).attr('id').split('-')[1]))
    });
    layers.each(function() {
      layerOrder.push(parseInt(d3.select(this).attr('id').split('-')[1]))
    });

    layerOrder.sort((a, b) => b - a);

    for (let i = 0; i < layerOrder.length - 1; i++) {
      const cursor = layerOrder[i];
      const currLayer = d3.select('#module-structure').selectAll('g').filter(function() { 
        const idTokens = d3.select(this).attr('id').split('-');
        return parseInt(idTokens[1]) == cursor && idTokens[2] == '0';
      });

      const prevLayer = d3.select('#module-structure').selectAll('g').filter(function() { 
        const idTokens = d3.select(this).attr('id').split('-');
        return parseInt(idTokens[1]) == layerOrder[i + 1] && idTokens[2] == '0';
      });

      currLayer.each(function() {
        const dstIR = d3.select(this);
        const layerClass = dstIR.attr('class');
        const dstImageIndex = dstIR.attr('id').split('-')[3];

        if (layerClass.includes('conv')) {
          prevLayer.each(function() {
            const srcIR = d3.select(this);
            const srcImageIndex = srcIR.attr('id').split('-')[3];
            addImageConnection(srcIR, dstIR, layerOrder[i + 1], cursor, srcImageIndex ,dstImageIndex);
          });
        }
        else if(layerClass === 'IntermediateResult-add' || layerClass === 'IntermediateResult-cat'  || layerClass.includes('pool')){
          const srcIR = prevLayer.filter(function() {
            return d3.select(this).attr('id').split('-')[3] === dstImageIndex;
          })
          addImageConnection(srcIR, dstIR, layerOrder[i + 1],cursor, dstImageIndex ,dstImageIndex);
        }
      });
    }
    pathData.forEach(path => {
      d3.select('#module-structure').append('path')
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
    const IRs = d3.select('#module-structure').selectAll('g').filter(function() { return this.getAttribute('class').includes('IntermediateResult') });
    let paths = undefined;
    
    IRs.on('mouseover', function() {
      const idTokens = d3.select(this).attr('id').split('-');
      const layerIndex = idTokens[1];
      const IRIndex = idTokens[3];
      const IRClass = d3.select(this).attr('class');

      paths = d3.select('#module-structure').selectAll('path').filter(function() {
        const edgeIndex = d3.select(this).attr('id').split('-');
        const isSrcLayerPath = (parseInt(edgeIndex[1]) === parseInt(layerIndex)) && (edgeIndex[2] === IRIndex);
        const isDstLayerPath = (edgeIndex[3] === layerIndex) && (edgeIndex[4] === IRIndex);

        return isDstLayerPath || isSrcLayerPath;
      });

      paths.attr('stroke-width', 3)
        .style('stroke-opacity', 0.7);

      
      const strokeFill = (IRClass.includes('relu')) ? 'black' : (IRClass.includes('bn')) ? 'black' : 'gray'

      d3.select(this).append('rect')
        .attr('class', 'IR-wrapper')
        .attr('width', imageWidth)
        .attr('height', imageHeight)
        .attr('fill', 'none')
        .attr('stroke', strokeFill)
        .attr('stroke-width', 3)
        .style('stroke-opacity', 1);

    }).on('mouseout', function() {
      paths.attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);

      d3.select(this).select('rect.IR-wrapper').remove();
    });
  }

  function setLinearLayerEvents(moduleLayer){
    const blocks = d3.select('#module-structure').selectAll('rect.block');
    //linear block event handling
    blocks.on('mouseover', function() {
      d3.select(this).style('stroke-width', 3);
    }).on('mouseout',function() {
      d3.select(this).style('stroke-width', 1);
    }).on('click', function(_, __, i) {
      d3.select('#module-structure').select('g.Prediction-result').remove();
      d3.select('#module-structure').selectAll('path').remove();
      pathData = [];

      const selectedBlock = d3.select(this);
      const linearLayerGroup = d3.select(this.parentNode);
      const selectedLayerDepth = parseInt(linearLayerGroup.attr('id').split('-')[1]);

      if(linearLayerGroup.attr('class') == 'IntermediateResult-softmax'){
        //Infobox Setting --> 인덱스에 따라서 모델 변경
        const infoBoxIndex = selectedBlock.attr('id').split('-')[1];
        
        const translateValues = linearLayerGroup.attr('transform').match(/translate\(([^)]+)\)/)[1];
        // Split the values into an array and convert them to numbers
        const [groupX, groupY] = translateValues.split(',').map(Number);

        const blockX = parseFloat(d3.select(this).attr('x'));
        const blockY = parseFloat(d3.select(this).attr('y'));
        
        const padding = { top: 40, right: 40, bottom: 40, left: 40 };
        const width = 600;
        const height = 300;        
        
        const predictionLabel = moduleLayer[0]['top5']; 
        // Scale for the bars
        const x = d3.scaleLinear()
                    .domain([0, 1])
                    .range([0, width - padding.left - padding.right]);
    
        const y = d3.scaleBand()
                    .domain(predictionLabel.map(d => imagenetClasses[d]))
                    .range([0, height - padding.top - padding.bottom])
                    .padding(0.1);
        
        const g = d3.select('#module-structure')
                    .append('g')
                    .attr('class', 'Prediction-result')
                    .attr('transform', `translate(${groupX + blockX + 150}, ${groupY + blockY})`);

        const gradient = g.append("defs")
                          .append("linearGradient")
                          .attr("id", "gradient")
                          .attr("x1", "0%")
                          .attr("x2", "100%")
                          .attr("y1", "0%")
                          .attr("y2", "0%");

        gradient.append("stop")
                .attr("offset", "0%")
                .attr("style", "stop-color:rgb(255,165,0);stop-opacity:1");

        gradient.append("stop")
                .attr("offset", "100%")
                .attr("style", "stop-color:rgb(255,215,0);stop-opacity:1");

        g.append('rect')
          .attr('class', 'Infobox')
          .attr('width', width)
          .attr('height',  height)
          .attr('fill', 'white')
          .attr('stroke', '#ccc')
          .attr('stroke-width', '1px');

        const maxTextWidth = width - padding.left - padding.right;

        // 막대 생성
        g.selectAll('.bar')
          .data(predictionLabel)
          .enter().append('rect')
          .attr('class', 'bar')
          .attr('x', padding.left)  // 막대의 x 시작 위치
          .attr('y', d => padding.top + y(imagenetClasses[d]) + y.bandwidth() / 2 + 45)  // 막대의 y 위치
          .attr('width', d => x(moduleLayer[0]['softmax_output'][d]))
          .attr('height', 10)  // 막대의 높이
          .attr('fill', 'url(#gradient)');

        // 레이블 추가
        g.selectAll('.label')
          .data(predictionLabel)
          .enter()
          .append('text')
          .attr('class', 'label')
          .attr('x', padding.left)  // 텍스트의 x 시작 위치
          .attr('y', d => padding.top + y(imagenetClasses[d]) + y.bandwidth() / 2 + 40)
          .attr('dy', '-0.35em')  // 텍스트를 수직으로 중앙에 정렬
          .attr('text-anchor', 'start')
          .text(d => truncateText(imagenetClasses[d], maxTextWidth))
          .style('fill', 'black');

        // 확률값 추가
        g.selectAll('.value')
          .data(predictionLabel)
          .enter()
          .append('text')
          .attr('class', 'value')
          .attr('x', d => padding.left + x(moduleLayer[0]['softmax_output'][d]) + 5)  // 값의 x 시작 위치
          .attr('y', d => padding.top + y(imagenetClasses[d]) + y.bandwidth() / 2 + 55)
          .attr('text-anchor', 'start')
          .text(d => `${(moduleLayer[0]['softmax_output'][d] * 100).toFixed(2)}%`)
          .style('fill', 'black');

          
        // 중앙에 가장 높은 확률의 항목 라벨 추가
        g.append("text")
          .attr("x", width / 2)
          .attr("y", padding.top - 10)
          .attr("text-anchor", "middle")
          .attr("font-size", "24px")
          .attr("fill", "black")
          .text(truncateText(imagenetClasses[infoBoxIndex], maxTextWidth));

        // 막대 생성
        g.append('rect')
          .attr('class', 'bar')
          .attr('x', padding.left)  // 막대의 x 시작 위치
          .attr('y', padding.top + 20)  // 막대의 y 위치
          .attr('width', x(moduleLayer[0]['softmax_output'][infoBoxIndex]))
          .attr('height', 20)  // 막대의 높이
          .attr('fill', 'url(#gradient)');

        g.append('text')
          .attr('class', 'value')
          .attr("x", width / 2)
          .attr("y", padding.top + 15)
          .attr("text-anchor", "middle")
          .attr("font-size", "24px")
          .text(d => `${(moduleLayer[0]['softmax_output'][infoBoxIndex] * 100).toFixed(3)}%`)
          .style('fill', 'black');
      }

      //select PrevLayer Rects
      if(selectedLayerDepth > 0){
        const prevBlocks = d3.select('#module-structure').selectAll('g').filter(function(){
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
            addBlockConnection(prevBlock, selectedBlock, selectedLayerDepth, prevBlockIndex, i);
        });
      }
      //select NextLayer Rects
      if(selectedLayerDepth < moduleLayerDepth){
        const nextBlocks = d3.select('#module-structure').selectAll('g').filter(function(){
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
            addBlockConnection(selectedBlock, nextBlock, selectedLayerDepth + 1, i, nextBlockIndex);
        });
      }

      pathData.forEach(path => {
      d3.select('#module-structure').append('path')
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
  function truncateText(text, maxWidth) {
    const context = document.createElement('canvas').getContext('2d');
    context.font = '24px sans-serif';
    let width = context.measureText(text).width;
    let ellipsis = '...';
    let ellipsisWidth = context.measureText(ellipsis).width;

    if (width <= maxWidth) {
      return text;
    }

    while (width >= maxWidth - ellipsisWidth && text.length > 0) {
      text = text.slice(0, -1);
      width = context.measureText(text).width;
    }

    return text + ellipsis;
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

  function addImageConnection(srcImage, dstImage, srcLayerIndex, dstLayerIndex, srcImageIndex, dstImageIndex){
    const srcTranslate = srcImage.attr('transform').match(/translate\(([^)]+)\)/);
    let srcCoordinates = srcTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    srcCoordinates[0] = srcCoordinates[0] + imageWidth;
    srcCoordinates[1] = srcCoordinates[1] + imageHeight / 2;

    const dstTranslate = dstImage.attr('transform').match(/translate\(([^)]+)\)/);
    let dstCoordinates = dstTranslate[1].split(',').map(function(d) { return parseFloat(d); });
    dstCoordinates[1] = dstCoordinates[1] + imageHeight / 2;

    pathData.push({ id:`edge-${srcLayerIndex}-${srcImageIndex}-${dstLayerIndex}-${dstImageIndex}`, source: srcCoordinates, target: dstCoordinates}) 
  }
  // Draw Conv Module 
  function drawConvModuleDetail(moduleLayers){
    let visibleLayerIndex = 0;
    let hiddenLayerCount = 0;
    let x;
    let y;
    console.log(moduleLayers)
    moduleLayers.forEach((layer, layerIndex) => {
      //Input Layer
      if(layerIndex === 0){   
        const inputX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const inputY = moduleYPadding + (offsetY);
        drawLayer(layer['input'], visibleLayerIndex, hiddenLayerCount, inputX, inputY, 'inline', 'input');

        visibleLayerIndex++;
        x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        y = moduleYPadding + (offsetY);
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type']);
      }
      //ReLU & BatchNorm Layer
      else if(layer['layer_type'] === 'relu' || layer['layer_type'] === 'bn'){
        hiddenLayerCount++;
        x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        y = moduleYPadding + (offsetY)
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['layer_type']);
      }
      //Other Layers (Conv, Pool)
      else if(layer['layer_type'] === 'conv' || layer['layer_type'].includes('pool')){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        y = moduleYPadding + (offsetY);
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type']);
      }
    });
    moduleLayerDepth = visibleLayerIndex;
  }

  // Draw Avgpool Module
  function drawAvgpoolModuleDetail(moduleLayers){
    moduleLayers.forEach((layer, layerIndex) => {
      //Input Layer 
      if(layerIndex === 0){

        const inputX = moduleXPadding + (layerIndex) * (imageWidth + offsetX);
        const inputY = moduleYPadding + (offsetY);
        console.log(`layerIndex: ${layerIndex}`)
        drawLayer(layer['input'], 0, 0, inputX, inputY, 'inline', 'input');

        const x = moduleXPadding + (layerIndex + 1) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY)
        drawLayer(layer['output'], layerIndex + 1, layerIndex, x, y, 'inline', layer['layer_type']);

        moduleLayerDepth = layerIndex + 1;
      }
      else if(layer['layer_type'] != 'flatten'){
        // Other Layers (Pool)
        const x = moduleXPadding + (layerIndex + 1) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY)
        drawLayer(layer['output'], layerIndex + 1, layerIndex, x, y, 'inline', layer['layer_type']);
        moduleLayerDepth = layerIndex + 1;
      }
    });
  }
  
  // Draw Linear Module
  function drawLinearModuleDetail(moduleLayers){
    let visibleLayerIndex = 0;
    let hiddenLayerCount = 0;
    console.log(moduleLayers)
    moduleLayers.forEach((layer, layerIndex) => {
      //Input Layer (Flatten input)
      if(layerIndex === 0){
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLinear(layer['input'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type']);
      }
      //Last Layer contains Top-10 prediction labes (output_index) and probability (output)
      if(layerIndex === (moduleLayers.length - 1)){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        //TODO(YSKIM): Print top 10 labels
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);

        drawLinear(layer['softmax_output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', 'softmax');
      }
      //ReLU Layer
      else if(layer['layer_type'] === 'relu'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLinear(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['layer_type']);
      }
      //Other Layers (Linear)
      else if(layer['layer_type'] === 'Linear'){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLinear(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type']);
      }
    });
    moduleLayerDepth = visibleLayerIndex;
  }

  // Draw Residual Module
  function drawResidualModuleDetail(moduleLayers){  
    let visibleLayerIndex = 0;
    let hiddenLayerCount = 0;
    let layer = undefined;

    const numLayers = moduleLayers['branches'][0].length + moduleLayers['layers'].length;
    for(let layerIndex = 0; layerIndex < numLayers; layerIndex++){
      layer = layerIndex < moduleLayers['branches'][0].length ? moduleLayers['branches'][0][layerIndex] : moduleLayers['layers'][layerIndex - moduleLayers['branches'][0].length];
      
      if(layerIndex == 0){
        const inputX = moduleXPadding;
        const inputY = moduleYPadding + (offsetY);
        drawLayer(layer['input'], 0, 0, inputX, inputY, 'inline', 'input');
      }
      if(layer['layer_type'] === 'conv' || layer['layer_type'].includes('pool') || layer['layer_type'] === 'add' ){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type']);
      }
      else if(layer['layer_type'] === 'relu' || layer['layer_type'] === 'bn'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding + (offsetY);

        visibleLayerIndex = visibleLayerIndex;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['layer_type']);
      }
    }
    moduleLayerDepth = visibleLayerIndex;
  }

  // Draw Inception Module
  function drawInceptionModuleDetail(moduleLayers){
    console.log(moduleLayers);
    let branchName = '';
    let hiddenLayerCount = 0;
    let visibleLayerIndex = 0;
    const detailSVG = d3.select('#module-structure');
    let branchDepth = 0;

    moduleLayers['branches'].forEach((branch, branchIndex) => {
      branchName = 'branch' + (branchIndex + 1).toString();
      branches.push(branchName);
      detailSVG.append('g')
      .attr('id', `${branchName}`)
      .attr('class', 'branch');
      //Input Layer 
      hiddenLayerCount = 0;
      visibleLayerIndex = 0;
      if(branchIndex === 0){
        const inputX = moduleXPadding;
        const inputY = moduleYPadding;
        drawLayer(branch[0]['input'], visibleLayerIndex, 0, inputX, inputY, 'inline', 'Input');
      }
      branch.forEach((layer) =>{
        const layerVisible = layer['layer_type'] === 'conv' || layer['layer_type'] === 'pool' ? true: false;
        if(layerVisible){
          visibleLayerIndex++;
          hiddenLayerCount = 0;
        }
        else{
          hiddenLayerCount++;
        }
        const layerX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const layerY = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, layerX, layerY, layerVisible, layer['layer_type'], branchName);
      });
      branchDepth = Math.max(branchDepth, visibleLayerIndex);
    });

    visibleLayerIndex = branchDepth;
    moduleLayers['layers'].forEach((layer) => {
      //Conv, Pool, cat Layer
      if(layer['layer_type'] === 'conv' || layer['layer_type'].includes('pool') || layer['layer_type'] === 'cat'){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const layerX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const layerY = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, layerX, layerY, 'inline', layer['layer_type']);
      }
      //ReLU & BatchNorm Layer
      else if(layer['layer_type'] === 'relu' || layer['layer_type'] === 'bn'){
        hiddenLayerCount++;
        const layerX = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const layerY = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, layerX, layerY, 'none', layer['layer_type']);
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

      d3.select('#module-structure').selectAll('g.IntermediateResult-cat').each(function(){
        const currentImageIndex = d3.select(this).attr('id').split('-')[3];
        const newImageIndex = `IR-${numLayerCurrentBranch + 1}-0-${currentImageIndex}`
        d3.select(this).attr('id', newImageIndex);
      });

      moduleLayerDepth = numLayerCurrentBranch + 1;

      d3.select('#module-structure').selectAll('path').remove();
      pathData = [];
      drawInceptionLayerConnections();
    }
  }

  $: if(selectedBranch) {
    updateInceptionBranch();
  }

  function drawLinear(layer, visibleLayerIndex, layerIndex, x, y, display = 'inline', layerClass){
    const [max, min] = getLayerMaxMin(layer);
    const boundaryValue = Math.max(Math.abs(min), Math.abs(max));
    const colorScale = d3.scaleLinear()
    .domain([-boundaryValue, 0, boundaryValue])
    .interpolate(d3.interpolate)
    .range([interpolateRdBu(0), interpolateRdBu(0.5), interpolateRdBu(1)]);

    const detailSVG = d3.select('#module-structure');
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

    drawLegend(x + (linearRectWidth - imageWidth)/2 , y + linearRectHeight * layer.length + offsetY, boundaryValue, display);
  }

  // Draw a layer which is contains IR results (Image or Linear)
  function drawLayer(layerImages, visibleLayerIndex, layerIndex, layerX, layerY, display, layerClass, branchName = 'none'){
    // Color Sacle of the layer images
    const [max, min] = getLayerMaxMin3D(layerImages);
    const boundaryValue = Math.max(Math.abs(min), Math.abs(max));
    
    console.log(max);
    console.log(min);

    const colorScale = d3.scaleLinear()
    .domain([-boundaryValue, 0, boundaryValue])
    .interpolate(d3.interpolate)
    .range([interpolateRdBu(0), interpolateRdBu(0.5), interpolateRdBu(1)]);

    let ImageX = 0;
    let ImageY = 0;
    layerImages.forEach((image, imageIndex) => {
      ImageX = layerX;
      ImageY = layerY + (imageHeight + offsetY) * (imageIndex);
      drawImage(image, visibleLayerIndex, layerIndex, imageIndex, colorScale, ImageX, ImageY, display, layerClass, branchName);
    });


    // If layers over the svg size, update svg size.
    const detailSVG = d3.select('#module-structure');
    const imageSize = 133;
    const currentWidth = detailSVG.attr('width');
    const currentHeight = detailSVG.attr('height');
    const requiredWidth = Math.max(currentWidth, ImageX + imageSize);
    const requiredHeight = Math.max(currentHeight, ImageY + imageSize);

    updateSVGSize(requiredWidth, requiredHeight);
    
    drawLegend(ImageX, ImageY + imageHeight + offsetY, boundaryValue, display);
  }

  function drawLegend(x, y, boundaryValue, display){
    const legendHeight = 10;
    const legendWidth = 133;

    const legendGroup = d3.select('#module-structure').append("g")
      .attr('class', 'legend-group')
      .attr("transform", `translate(${x}, ${y})`)
      .style('display', display);

    const gradient = legendGroup.append("defs")
      .append("linearGradient")
      .attr("id", "linear-gradient")
      .attr("x1", "0%")
      .attr("x2", "100%")
      .attr("y1", "0%")
      .attr("y2", "0%");

    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", interpolateRdBu(0));

    gradient.append("stop")
      .attr("offset", "50%")
      .attr("stop-color", interpolateRdBu(0.5));

    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", interpolateRdBu(1));

    legendGroup.append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#linear-gradient)");

    legendGroup.append("text")
      .attr("x", 0)
      .attr("y", legendHeight + 15)
      .attr("text-anchor", "start")
      .text((-boundaryValue).toFixed(2));

    legendGroup.append("text")
      .attr("x", legendWidth)
      .attr("y", legendHeight + 15)
      .attr("text-anchor", "end")
      .text(boundaryValue.toFixed(2));
}

  //Draw an Image
  function drawImage(image, visibleLayerIndex, layerIndex, imageIndex, colorScale, x, y, display, layerClass, branchName = 'none') {
    const cellSize = 133 / image.length;
    const numRows = image.length;
    const numCols = image[0].length; 
    const strokeFill = (layerClass === 'relu') || (layerClass === 'bn') ? 'black' : 'gray'
    // const strokeWidth = 1;
    const strokeWidth = 1;
    const className = (branchName === 'none') ? `IntermediateResult-${layerClass}`:  `IntermediateResult-${branchName}-${layerClass}`
    let detailSVG = d3.select('#module-structure');
    
    if(branchName !== 'none'){
      detailSVG = d3.select('#module-structure').select(`g#${branchName}`);
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
    let detailSVG = d3.select('#module-structure');
    if(typeof selectedBranch !== 'undefined'){
      detailSVG = d3.select(`g#${selectedBranch}`);
    }
    detailSVG.selectAll('g.IntermediateResult-relu').each(function() {
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
    let detailSVG = d3.select('#module-structure');
    let bnClassSelector = 'g.IntermediateResult-bn';
    if(typeof selectedBranch !== 'undefined'){
      detailSVG = d3.select('#module-structure').select(`g#${selectedBranch}`);
      bnClassSelector = `g.IntermediateResult-${selectedBranch}-bn`;
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
  #module-container {
    overflow: hidden;
    height: 100%;
    width: 100%;
  }
  #model-svg {
    cursor: grab;
  }
  #model-svg:active {
    cursor: grabbing;
  }
  #module-svg {
    cursor: grab;
  }
  #module-svg:active {
    cursor: grabbing;
  }

  
</style>

<Header/>

<Container fluid>
  <Row class="h-100" style="height: calc(100vh - 60px);">
    <Col style="max-width: 260px; height: calc(100vh - 60px); overflow-y: auto; border: 1px solid rgba(225,225,225,255); background-color: rgba(249,249,249,255); padding: 10px; font-family: Arial, sans-serif;">
      <Row style="margin-bottom: 5px; align-items: center; justify-content: flex-end; width: 100%;">
        <FormCheck type="switch" id="form-model" label="Hugging Face Model URL" bind:checked={isHuggingFaceModel} />
      </Row>
      <Row style="margin-bottom: 5px; align-items: center;">
        <Col>
        <FormGroup>
          <Label for="model-select">Model</Label>
        </FormGroup>
        </Col>
        <Col>
        <FormGroup>
          <Input type="select" bind:value={selectedModel} id="model-select" disabled={isHuggingFaceModel}>
            {#each imagenetModels as modelName}
              <option value={modelName}>{modelName}</option>
            {/each}
          </Input>
        </FormGroup>
      </Col>
      </Row>
      <Row style="margin-bottom: 5px; align-items: center;">
        <FormGroup>
          <Label for="HuggingFace-url" class="me-2 mb-0">URL</Label>
        </FormGroup>
        <FormGroup>
          <Input type="text" id="model-url" placeholder="Type 'microsoft/resnet-18' here" disabled={!isHuggingFaceModel} />
        </FormGroup>
      </Row>
      <Row style="margin-bottom: 5px; align-items: center; justify-content: flex-end; width: 100%;">
        <FormCheck type="switch" id="form-model" label="User Image Input" bind:checked={isUserInputImage} />
      </Row>
      <Row style="margin-bottom: 5px; align-items: center;">
        <FormGroup>
          <Label for="class-select">Class</Label>
        </FormGroup>
        <FormGroup>
          <Input type="select" bind:value={selectedClass} id="class-select" disabled={isUserInputImage}>
            {#each Object.entries(imagenetClasses) as [index, className]}
              <option value={index}>{index}: {className}</option>
            {/each}
          </Input>
        </FormGroup>
      </Row>

      <Row style="margin-bottom: 5px; align-items: center;">
        <FormGroup>
          <Label for="class-select" class="me-2 mb-0">Image Input</Label>
        </FormGroup>
        <FormGroup>
          <Input type="file" id="image-upload" accept="image/*" on:change={handleFileChange} disabled={!isUserInputImage} />
        </FormGroup>
      </Row>
    
      <Row style="margin-bottom: 5px; align-items: center;">
        <Button class="text-bg-primary" style="width: 100%; margin-top: 10px;" on:click={loadModelView}>
          Load Model
        </Button>
      </Row>
      <Row style="margin-bottom: 5px; align-items: center;">
        <div class="justify-content-end">
          <div class="switch-container align-items-center" style="gap: 10px;">
            <FormCheck type="switch" id="form-ReLU" label="ReLU" bind:checked={reluActive} on:change={toggleReLU} disabled={!openModal} />
          </div>
          <div class="switch-container align-items-center" style="gap: 10px;">
            <FormCheck type="switch" id="form-BN" label="BatchNorm" bind:checked={batchNormActive} on:change={toggleBN} disabled={!openModal} />
          </div>
        </div>
        <div class="justify-content-end"> 
          {#if selectedModule == "inception" && selectedBranch != undefined}
            <div>
              <Input type="select" bind:value={selectedBranch} id="branch-select" style="width: auto;">
                {#each branches as branch}
                  <option value={branch}>{branch}</option>
                {/each}
              </Input>
            </div>
          {/if}
        </div>
      </Row>
      <Row class="mt-auto">
      </Row>
    </Col>
    <Col class="p-0" style="flex-grow: 1; height: calc(100vh - 60px);">
      <Row class="m-0" style="width: 100%; height: calc(50vh - 30px); border: 1px solid rgba(225,225,225,255);">
      <div class="p-0" id="model-container" style="width: 100%; height: 100%; overflow: auto;">
        <svg id="model-svg" style="width: 100%; height: 100%; display: block; min-width: 100%; min-height: 100%;">
          <g id="model-structure"></g>
        </svg>
      </div>
      </Row>
      <Row class="m-0" style="width: 100%; height: calc(50vh - 30px); border: 1px solid rgba(225,225,225,255);">
          <div id="module-container">
            <svg id="module-svg" style="width: 100%; height: 100%; display: block; min-width: 100%; min-height: 100%;">
              <g id="module-structure"></g>
            </svg>
          </div>
      </Row>
    </Col>
  </Row>
</Container>