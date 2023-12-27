<script>
	import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { construct_svelte_component } from 'svelte/internal';

	// import Overview from './Overveiw.svelte'
	// import Conv from '.operators/Convolutional.svelte'

  //######################################################################//
  let selectedModel = 'resnet18';
  let moduleLists = undefined;
  let focusedModule = undefined;
  let modelData = undefined;
  let detail_svg = undefined;
  onMount(() => {
    detail_svg = d3.select('#IR-detail-load')
                    .append('svg')
                    .attr('width', 2000)
                    .attr('height', 1000);  
  });

  const moduleXMargin = 10;
  const moduleYMargin = 200;
  const moduelWidth = 100;
  const moduleHeight = 400;
  let moduleStruct = undefined;
  let moduleNum = undefined;

  async function loadModelView() {

    console.log(selectedModel)
    moduleStruct = [];

    const response = await fetch(`/output/${selectedModel}_info.json`);
    modelData = await response.json();
    console.log("Loaded JSON data:", modelData);

    // JSON 객체의 모든 키를 출력합니다.
    console.log("Keys in JSON:", Object.keys(modelData));

    const uniqueModules = new Set(); // 중복을 허용하지 않는 Set 생성
    // structure column
    for (const key in modelData) {
      const layer = data[key];
      if (module.module_index !== moduleIndex) {
        // moduleStruct.push(module.module_type);
        moduleIndex = module.module_index;
      }
    }
    
    moduleStruct.push('conv');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('residual');
    moduleStruct.push('avgpool');
    moduleStruct.push('linear');


    moduleStruct.forEach((moduleName, moduleIndex) => {
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


      let moduleGroup = detail_svg.append('g')
          .on('mouseover', function() {
            d3.select(this).select('rect')
              .transition()
              .duration(250)
              .style('stroke-width', 3);

            // 텍스트에도 스타일 변경 적용
            d3.select(this).select('text')
              .transition()
              .duration(250)
              .style('font-weight', 600); // 예시: 텍스트 색상을 빨간색으로 변경
          })
          .on('mouseout', function() {
            d3.select(this).select('rect')
              .transition()
              .duration(250)
              .style('stroke-width', 0);

            // 텍스트 스타일을 원래대로 복원
            d3.select(this).select('text')
              .transition()
              .duration(250)
              .style('font-weight', 300); // 예시: 텍스트 색상을 다시 검은색으로 변경
          }).on('click', function(){      
              // 이미 클릭된 상태라면 해제
              if (this.id === clickedOverlayRect) {
                  d3.select(this).style('fill', 'none');
                  clickedOverlayRect = null;
                  return;
              }

              // 이전에 클릭된 rect가 있다면, 그 스타일을 해제
              if (clickedOverlayRect) {
                d3.select(`#${clickedOverlayRect}`).style('fill', 'none');
              }

              // 현재 rect의 스타일 설정 및 추적
              d3.select(this).style('fill', 'gray');
              clickedOverlayRect = this.id;
          });
           
      moduleGroup.append('rect')
          .attr('x', moduleIndex * (moduelWidth + moduleXMargin))
          .attr('y', moduleYMargin)
          .attr('width', moduelWidth)
          .attr('height', moduleHeight)
          .attr('fill', moduleFills)
          .style('stroke-width', 0)
          .style('stroke', 'gray');

      moduleGroup.append('text')
          .attr('x', moduleIndex * (moduelWidth + moduleXMargin) + moduelWidth / 2)
          .attr('y', moduleYMargin + moduleHeight / 2)
          .attr('text-anchor', 'middle') 
          .attr('dominant-baseline', 'middle') 
          .text(moduleName);
    });
  }

  async function loadJSON() {
    console.log(selectedModel)

    const response = await fetch(`/output/${selectedModel}_info.json`);
    modelData = await response.json();
    console.log("Loaded JSON data:", modelData);

    // JSON 객체의 모든 키를 출력합니다.
    console.log("Keys in JSON:", Object.keys(modelData));

    for (const key in modelData){
        console.log(key);
        uniqueModules.add(modelData[key]['module']); // module 값 추가

        drawLayer(modelData[key]['output'])
        break
    }
    // moduleLists = Array.from(modules);
  }

  function drawImage(image, max, min, offsetX, offsetY) {
    const cellSize = 1;
    const g = detail_svg .append('g')
      .attr('transform', `translate(${offsetX}, ${offsetY})`);
    
    console.log(image)
    image.forEach((row, i) => {
      row.forEach((value, j) => {
        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', getFillColor(value));
      });
    });
  }
  // TODO(YSKIM): LayerImage대신 Layerindex를 쓰는걸로
  function drawLayer(layerImages, offsetX, offestY) {
    const [max, min] = getLayerMaxMin(layerImages);
    const imageHeight = 113;
    const padding = 10;

    layerImages.forEach((image, index) => {
      const offsetX = 10;
      const offsetY = index * (imageHeight + padding);
      drawImage(image, max, min, offsetX, offsetY);
    });
  }

  // 배열의 각 요소를 그레이스케일 색상으로 변환하는 함수
  function getFillColor(value) {
    const color = Math.floor(value/255);
    return `rgb(${color},${color},${color})`;
  }

  // TODO(YSKIM): Module Max Min으로 바꿔야함
  function getLayerMaxMin(layerImages) {
    const flatImages = layerImages.flat(3);

    const sortedImages = flatImages.sort((a, b) => a - b);

    const max = Math.max(...flatImages);
    const min = Math.min(...flatImages);
    // console.log(max)
    // console.log(min)
    
    return [max, min];
  }

  function getImageMaxMin(image) {
    const flatImage = image.flat(2);

    const max = Math.max(...flatImage);
    const min = Math.min(...flatImage);

    return [max, min];
  }

  function normalizeAndScale(image, max, min) {
    return image.map(row =>
      row.map(value => (value - min) / (max - min) * 255)
    );
  }
  /*
  function drawModule() {
    for (moduleType in moduleLists) {
      if (moduleType === 'conv') {
        //색상 파랑
      } 
      else if (moduleType === 'dense') {
        //색상 노랑
      }
      else if (moduleType === 'inception') {
        //색상 회색
      }
      else if (moduleType === 'residual') {
        //색상 녹색
      }
    }
  }
  */

</script>

<select bind:value={selectedModel}>
  <option value='resnet18'>ResNet</option>
  <option value='vgg16'>VGG</option>
  <option value='alexnet'>AlexNet</option>
  <option value='googlenet'>GoogleNet</option>
</select>
<button on:click={loadModelView}>Load</button>

<div id='model-container'></div>  
<div id='model-load'></div>
<div id='IR-detail-load'></div>

<!-- <div id='overview'>
	<Overview />
</div> -->