// --- State Management ---
let model, lossFn, optimizer;
let trainingData = { X: null, y: null };
let lossHistory = [];
let chart;
let isTraining = false;
let currentEpoch = 0;

// --- DOM Elements ---
const canvas = document.getElementById('network-canvas');
const ctx = canvas.getContext('2d');
const lossCanvas = document.getElementById('loss-chart');
const trainBtn = document.getElementById('train-btn');
const resetBtn = document.getElementById('reset-btn');

const neuronsInput = document.getElementById('neurons');
const lrInput = document.getElementById('lr');
const regInput = document.getElementById('reg');
const epochsInput = document.getElementById('epochs');

const neuronsVal = document.getElementById('neurons-val');
const lrVal = document.getElementById('lr-val');
const regVal = document.getElementById('reg-val');

const epochDisplay = document.getElementById('epoch-display');
const lossDisplay = document.getElementById('loss-display');
const accDisplay = document.getElementById('acc-display');

// --- Initialization ---
function init() {
    setupInputs();
    setupChart();
    generateData();
    resetModel();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function setupInputs() {
    neuronsInput.addEventListener('input', (e) => {
        neuronsVal.textContent = e.target.value;
        resetModel();
    });
    lrInput.addEventListener('input', (e) => {
        lrVal.textContent = e.target.value;
        if (optimizer) optimizer.lr = parseFloat(e.target.value);
    });
    regInput.addEventListener('input', (e) => {
        regVal.textContent = e.target.value;
        if (optimizer) optimizer.reg = parseFloat(e.target.value);
    });

    trainBtn.addEventListener('click', toggleTraining);
    resetBtn.addEventListener('click', resetModel);
}

function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
    drawNetwork();
}

function setupChart() {
    chart = new Chart(lossCanvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#30363d' },
                    ticks: { color: '#8b949e' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8b949e' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// --- Data Generation ---
function generateData(n = 200) {
    // Generate 3 blobs in 2D for visualization simplicity
    const centers = [[-2, -2], [2, 2], [-2, 2]];
    const X = [];
    const y = [];

    for (let c = 0; c < centers.length; c++) {
        for (let i = 0; i < n / centers.length; i++) {
            X.push([
                centers[c][0] + (Math.random() - 0.5) * 2,
                centers[c][1] + (Math.random() - 0.5) * 2
            ]);
            let oh = [0, 0, 0];
            oh[c] = 1;
            y.push(oh);
        }
    }

    trainingData.X = Matrix.fromArray(X);
    trainingData.y = Matrix.fromArray(y);
}

// --- Model Management ---
function resetModel() {
    isTraining = false;
    trainBtn.textContent = 'Start Training';
    currentEpoch = 0;
    lossHistory = [];

    const hidden = parseInt(neuronsInput.value);
    const lr = parseFloat(lrInput.value);
    const reg = parseFloat(regInput.value);

    model = new Sequential([
        new Linear(2, hidden),
        new ReLU(),
        new Linear(hidden, 3)
    ]);

    lossFn = new CrossEntropyLoss();
    optimizer = new SGD(model.layers, lr, reg);

    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();

    updateStats(0, 0);
    drawNetwork();
}

// --- Training Loop ---
function toggleTraining() {
    isTraining = !isTraining;
    trainBtn.textContent = isTraining ? 'Pause Training' : 'Resume Training';
    if (isTraining) {
        requestAnimationFrame(trainStep);
    }
}

function trainStep() {
    if (!isTraining || currentEpoch >= parseInt(epochsInput.value)) {
        if (currentEpoch >= parseInt(epochsInput.value)) {
            isTraining = false;
            trainBtn.textContent = 'Training Finished';
        }
        return;
    }

    // Perform one epoch (simplified as full batch for visual smoothness)
    const scores = model.forward(trainingData.X);
    const loss = lossFn.loss(scores, trainingData.y);
    const dloss = lossFn.backward(trainingData.y);
    model.backward(dloss);
    optimizer.step();

    currentEpoch++;

    // Calculate Accuracy
    let correct = 0;
    const probs = lossFn.probs.data;
    const targets = trainingData.y.data;
    for (let i = 0; i < trainingData.X.rows; i++) {
        let maxIdx = 0;
        let pMax = -1;
        for (let j = 0; j < 3; j++) {
            if (probs[i * 3 + j] > pMax) {
                pMax = probs[i * 3 + j];
                maxIdx = j;
            }
        }
        if (targets[i * 3 + maxIdx] === 1) correct++;
    }
    const acc = correct / trainingData.X.rows;

    // Update UI
    updateStats(loss, acc);

    if (currentEpoch % 1 === 0) {
        chart.data.labels.push(currentEpoch);
        chart.data.datasets[0].data.push(loss);
        if (chart.data.labels.length > 50) {
            // Keep chart readable
            // chart.data.labels.shift();
            // chart.data.datasets[0].data.shift();
        }
        chart.update('none'); // Update without animation for speed
        drawNetwork();
    }

    requestAnimationFrame(trainStep);
}

function updateStats(loss, acc) {
    epochDisplay.textContent = currentEpoch;
    lossDisplay.textContent = loss.toFixed(4);
    accDisplay.textContent = (acc * 100).toFixed(1) + '%';
}

// --- Visualization ---
function drawNetwork() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layers = [2, parseInt(neuronsInput.value), 3];
    const nodeRadius = 12;
    const layerSpacing = canvas.width / (layers.length + 1);

    const layerPos = layers.map((count, i) => {
        const x = layerSpacing * (i + 1);
        const ySpacing = canvas.height / (count + 1);
        return Array.from({ length: count }, (_, j) => ({
            x: x,
            y: ySpacing * (j + 1)
        }));
    });

    // Draw Connections
    for (let i = 0; i < layerPos.length - 1; i++) {
        const currLayer = layerPos[i];
        const nextLayer = layerPos[i + 1];

        // Find corresponding weight matrix
        // Sequential has: Linear (0), ReLU (1), Linear (2)
        const weightLayerIdx = i === 0 ? 0 : 2;
        const weights = model.layers[weightLayerIdx].params.W.data;
        const wRows = model.layers[weightLayerIdx].params.W.rows;
        const wCols = model.layers[weightLayerIdx].params.W.cols;

        for (let n1 = 0; n1 < currLayer.length; n1++) {
            for (let n2 = 0; n2 < nextLayer.length; n2++) {
                const w = weights[n1 * wCols + n2];
                const alpha = Math.min(Math.abs(w) * 5, 1); // Scale for visibility
                ctx.beginPath();
                ctx.moveTo(currLayer[n1].x, currLayer[n1].y);
                ctx.lineTo(nextLayer[n2].x, nextLayer[n2].y);
                ctx.strokeStyle = w > 0 ? `rgba(63, 185, 80, ${alpha})` : `rgba(248, 81, 73, ${alpha})`;
                ctx.lineWidth = Math.abs(w) * 2;
                ctx.stroke();
            }
        }
    }

    // Draw Nodes
    layerPos.forEach((nodes, i) => {
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = varColor('--sidebar-bg');
            ctx.fill();
            ctx.strokeStyle = varColor('--accent-color');
            ctx.lineWidth = 2;
            ctx.stroke();

            // Glow effect
            ctx.shadowBlur = 10;
            ctx.shadowColor = varColor('--accent-glow');
        });
    });
}

function varColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

init();
