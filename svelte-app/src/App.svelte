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
  let stepBeforeLine = undefined;
  let stepAfterLine = undefined;
  let straightLine = undefined;

  const branches = ['branch1','branch2','branch3','branch4'];
  const imagenetModels = ['alexnet', 'vgg16', 'googlenet', 'resnet18'];
  let imagenetClasses ={}
  onMount(async () => {
    const response = await fetch('/imageClasses.json');
    imagenetClasses = await response.json();
    selectedModel = imagenetModels[0];
    selectedClass = "0";
    modelSVG = d3.select('#model-container').select('svg');
    const zoom = d3.zoom()
      .scaleExtent([0.5, 2])  // zoom range
      .on('zoom', (event) => {
        modelSVG.select('g#model-structure').attr('transform', event.transform);
      });

    modelSVG.call(zoom);

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

  let infoBoxIndex = -1;
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
    d3.select('#module-svg')
      .attr('width', newWidth)
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

            console.log('test point');
    
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
    // selectedModule = selectedModuleName;
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
    d3.select('#module-svg').selectAll("*").remove();
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
      setLinearLayerEvents();
    }
    else if (selectedModuleInfo['type'] === 'residual'){
      drawResidualModuleDetail(selectedModuleInfo);
      drawLayerConnections();
      drawShortCuts();
      setLayerEvents();
    }
    else if (selectedModuleInfo['type'] === 'inception'){
      drawInceptionModuleDetail(selectedModuleInfo);
      drawLayerConnections();
      setLayerEvents();
    }
  }

  function drawShortCuts(){
    const residualLayer = d3.select('#module-svg').selectAll('g.IntermediateResult-add');

    residualLayer.each(function() {
      const dstIR = d3.select(this);
      const idTokens = dstIR.attr('id').split('-');
      const dstLayerIndex = idTokens[1];
      const dstImageIndex = idTokens[3];
      const srcIR = d3.select('#module-svg').select(`g#IR-0-0-${dstImageIndex}`);

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
        
      d3.select('#module-svg').append('path')
        .attr('d', stepBeforeLine(pathDataBeforeLine))
        .attr('fill', 'none')
        .attr('stroke', 'gray')
        .attr('class','residual-edge')
        .attr('id', `edge-${dstLayerIndex}-${dstImageIndex}-${dstImageIndex}`)
        .attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);
        
        d3.select('#module-svg').append('path')
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
      const currLayer = d3.select('#module-svg').selectAll('g').filter(function() { 
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

    const prevLayer = d3.select('#module-svg').selectAll('g').filter(function() { 
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
            addImageConnection(srcIR, dstIR, cursor, srcImageIndex ,dstImageIndex);
          });
        }
        else if(layerClass === 'IntermediateResult-Concat'|| layerClass === 'IntermediateResult-add' || layerClass.includes('pool')){  //ToDo(YSKIM): residual -> Add 
          const srcIR = prevLayer.filter(function() {
            return d3.select(this).attr('id').split('-')[3] === dstImageIndex;
          })
          addImageConnection(srcIR, dstIR, cursor, dstImageIndex ,dstImageIndex);
        }
      });
    }
    pathData.forEach(path => {
      d3.select('#module-svg').append('path')
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
    const IRs = d3.select('#module-svg').selectAll('g').filter(function() { return this.getAttribute('class').includes('IntermediateResult') });
    let paths = undefined;
    
    IRs.on('mouseover', function() {
      const idTokens = d3.select(this).attr('id').split('-');
      const layerIndex = idTokens[1];
      const IRIndex = idTokens[3];
      const IRClass = d3.select(this).attr('class');

      // const paths = d3.select('#module-svg').selectAll('path').filter(function() {
      paths = d3.select('#module-svg').selectAll('path').filter(function() {
        const edgeClass = d3.select(this).attr('class');
        const edgeIndex = d3.select(this).attr('id').split('-');
        const isNextLayerPath = (parseInt(edgeIndex[1]) - 1 === parseInt(layerIndex)) && (edgeIndex[2] === IRIndex);
        const isPrevLayerPath = (edgeIndex[1] === layerIndex) && (edgeIndex[3] === IRIndex);
        const isNextLayerNotResidual = (parseInt(edgeIndex[1]) - 1 === parseInt(layerIndex)) && (edgeClass === 'edge');

        return (isNextLayerPath && isNextLayerNotResidual) || isPrevLayerPath;
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
      const layerIndex = d3.select(this).attr('id').split('-')[1];
      const IRIndex = d3.select(this).attr('id').split('-')[3];

      paths.attr('stroke-width', 1)
        .style('stroke-opacity', 0.5);

      d3.select(this).select('rect.IR-wrapper').remove();
    });
  }

  function setLinearLayerEvents(){
    const softmaxBlocks = d3.select('#module-svg').select('g.Intermediate-softmax').selectAll('rect.block');
    const blocks = d3.select('#module-svg').selectAll('rect.block');


    //softmax block event handling    
    softmaxBlocks.on('mouseover', function() {
      const hoveredLabelIndex = d3.select(this).attr('id').split('-')[1];
      // hoveredsoftmaxLabel = {class:prob}
      // tooltipVisible = true;
    }).on('mouseout', function() {
      hoveredsoftmaxBlock = undefined;
      // tooltipVisible = false;
    });

    //linear block event handling
    blocks.on('mouseover', function() {
      d3.select(this).style('stroke-width', 3);
    }).on('mouseout',function() {
      d3.select(this).style('stroke-width', 1);
    }).on('click', function() {
      d3.select('#module-svg').selectAll('path').remove();
      pathData = [];

      const selectedBlock = d3.select(this);
      const selectedLayerDepth = parseInt(d3.select(this.parentNode).attr('id').split('-')[1]);
      const selectedBlockIndex = selectedBlock.attr('id').split('-')[1];

      //Infobox Setting --> 인덱스에 따라서 모델 변경
      infoBoxIndex = parseInt(selectedBlockIndex);
      // const selectedRect = this.getBoundingClientRect();

      //select PrevLayer Rects
      if(selectedLayerDepth > 0){
        const prevBlocks = d3.select('#module-svg').selectAll('g').filter(function(){
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
        const nextBlocks = d3.select('#module-svg').selectAll('g').filter(function(){
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
      d3.select('#module-svg').append('path')
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
        
        softmaxProbs = layer['softmax_output'];
        top5Index = layer['top5'];

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

    console.log(moduleLayers);

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
    // console.log(moduleLayers)
    // moduleLayers.forEach((layer, layerIndex) => {
    //   //last RelU Layer includes identity
    //   if(layerIndex === (moduleLayers.length - 1)){   
    //     const inputX = moduleXPadding;
    //     const inputY = moduleYPadding + (offsetY);
    //     drawLayer(layer['identity'], 0, 0, inputX, inputY, 'inline', 'input');

    //     visibleLayerIndex++;
    //     hiddenLayerCount = 0;
    //     const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
    //     const y = moduleYPadding + (offsetY);
    //     // const x1 = inputX + imageWidth/2;
    //     // const x2 = x + imageWidth/2;
    //     // drawShortcut(x1, x2, y, 1);
    //     drawLayer(layer['input'], visibleLayerIndex, 0, x, y, 'inline', 'Residual');
    //     drawLayer(layer['output'], visibleLayerIndex, 1, x, y, 'none', layer['layer_type']);
    //   }
    //   else if(layerNames[layerIndex].includes('downsample')){
    //     //ToDo: Add downsampling IR result
    //   } 
    //   else if(layer['layer_type'] === 'relu' || layer['layer_type'] === 'bn'){
    //     hiddenLayerCount++;
    //     const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
    //     const y = moduleYPadding + (offsetY);

    //     visibleLayerIndex = visibleLayerIndex;
    //     drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['layer_type']);
    //   }
    //   else if(layer['layer_type'] === 'conv' || layer['layer_type'].includes('pool')){
    //     visibleLayerIndex++;
    //     hiddenLayerCount = 0;
    //     const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
    //     const y = moduleYPadding + (offsetY);
    //     drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type']);
    //   }
    // });
    moduleLayerDepth = visibleLayerIndex;
  }

  // Draw Inception Module
  function drawInceptionModuleDetail(moduleLayers){
    console.log(moduleLayers);
    // branchLayerNum = getBranchStruct()
    let branchName = '';
    let hiddenLayerCount = 0;
    let visibleLayerIndex = 0;
    const detailSVG = d3.select('#module-svg');
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
      if(layer['layer_type'] === 'relu' || layer['layer_type'] === 'bn'){
        hiddenLayerCount++;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'none', layer['layer_type'], branchName);
      }
      //Other Layers (Conv, Pool)
      else if(layer['layer_type'] === 'conv' || layer['layer_type'].includes('pool')){
        visibleLayerIndex++;
        hiddenLayerCount = 0;
        const x = moduleXPadding + (visibleLayerIndex) * (imageWidth + offsetX);
        const y = moduleYPadding;
        drawLayer(layer['output'], visibleLayerIndex, hiddenLayerCount, x, y, 'inline', layer['layer_type'], branchName);
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

      d3.select('#module-svg').selectAll('g.IntermediateResult-Concat').each(function(){
        const currentImageIndex = d3.select(this).attr('id').split('-')[3];
        const newImageIndex = `IR-${numLayerCurrentBranch + 1}-0-${currentImageIndex}`
        d3.select(this).attr('id', newImageIndex);
      });

      moduleLayerDepth = numLayerCurrentBranch + 1;

      d3.select('#module-svg').selectAll('path').remove();
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

    const detailSVG = d3.select('#module-svg');
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
    const detailSVG = d3.select('#module-svg');
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
    const strokeFill = (layerClass === 'relu') ? 'black' : (layerClass === 'bn') ? 'black' : 'gray'
    // const strokeWidth = 1;
    const strokeWidth = 1;
    const className = (branchName === 'none') ? `IntermediateResult-${layerClass}`:  `IntermediateResult-${branchName}-${layerClass}`
    let detailSVG = d3.select('#module-svg');
    
    if(branchName !== 'none'){
      detailSVG = d3.select('#module-svg').select(`g#${branchName}`);
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
    let detailSVG = d3.select('#module-svg');
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
    let detailSVG = d3.select('#module-svg');
    let bnClassSelector = 'g.IntermediateResult-bn';
    if(typeof selectedBranch !== 'undefined'){
      detailSVG = d3.select('#module-svg').select(`g#${selectedBranch}`);
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
    overflow: auto;
    height: 100%;
    width: 100%;
  }
  #model-svg {
    cursor: grab;
  }
  #model-svg:active {
    cursor: grabbing;
  }
</style>

<Header/>

<Container fluid>
  <Row class="h-100" style="height: calc(100vh - 60px);">
    <Col class="d-flex flex-column" style="flex: 0 0 400px; max-width: 400px; height: calc(100vh - 60px); overflow-y: auto; border: 1px solid rgba(225,225,225,255); background-color: rgba(249,249,249,255);">
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
        <div class="d-flex justify-content-end">
          <div class="switch-container d-flex align-items-center">
            <FormCheck type="switch" id="form-ReLU" label="ReLU" bind:checked={reluActive} on:change={toggleReLU} disabled={!openModal} />
            <FormCheck type="switch" id="form-BN" label="BatchNorm" bind:checked={batchNormActive} on:change={toggleBN} disabled={!openModal} />
          </div>
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
        {#if openModal}
          <div>
            {#if selectedModule == "inception"}
              <div class="d-flex">
                <Input type="select" bind:value={selectedBranch} id="branch-select" class="me-3" style="width: auto;">
                  {#each branches as branch}
                    <option value={branch}>{branch}</option>
                  {/each}
                </Input>
              </div>
            {/if}
          </div>
          <div id="module-container">
            <svg id="module-svg">
            </svg>
          </div>
        {/if}
      </Row>
    </Col>
  </Row>
</Container>

{#if infoBoxIndex != -1}
  <div id="info-box" style="position: absolute; background: white; border: 1px solid black; padding: 10px;">
    <p>Name: ?</p>
    <p>Role: ?</p>
  </div>
{/if}
<!-- 
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
    <div id="module-container">
      <svg id="module-svg">
      </svg>
    </div>
  </ModalBody>
  <ModalFooter class="d-flex justify-content-end">
    <div class="switch-container d-flex align-items-center">
      <FormCheck type="switch" id="form-ReLU" label="ReLU" bind:checked={reluActive} on:change={toggleReLU} />
      <FormCheck type="switch" id="form-BN" label="BatchNorm" bind:checked={batchNormActive} on:change={toggleBN} />
    </div>
  </ModalFooter>
</Modal> -->

<!-- <div id='overview'>
	<Overview />
</div> -->