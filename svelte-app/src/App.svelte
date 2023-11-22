<script>
	import { onMount } from 'svelte';
	// import Board from './Board.svelte'

	let nn_weight = {'dummy': 'True'};
	onMount(async () => {
		try {
			// Replace this URL with the actual endpoint of your Flask server
			const response = await fetch('./layer');

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			const data = await response.json();
			console.log('Layer Data:', data);
		} catch (error) {
			console.error('Error fetching layer data:', error);
		}
	});
	
	// let session;
	// onMount(async () => {
	// 	// Create an ONNX session
	// 	session = new onnx.InferenceSession();

	// 	// Load the ONNX model
	// 	try {
	// 		await session.loadModel('./weight/resnet18_0_weights.onnx');
	// 		console.log('Model loaded successfully');
	// 		// Add your visualization logic here
	// 	} catch (err) {
	// 		console.error('Failed to load the ONNX model', err);
	// 	}
	// });
	let rand = -1;

	function getRand() {
		fetch('./rand')
			.then(d => d.text())
			.then(d => (rand = d));
	};

</script>

<h1>Your number is {rand}!</h1>
<button on:click={getRand}>Get a random number</button>