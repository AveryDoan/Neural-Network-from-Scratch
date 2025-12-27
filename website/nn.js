/**
 * Simple Matrix utility class for Linear Algebra operations
 */
class Matrix {
    constructor(rows, cols, data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Float64Array(rows * cols);
    }

    static fromArray(arr) {
        let m = new Matrix(arr.length, arr[0].length);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i * m.cols + j] = arr[i][j];
            }
        }
        return m;
    }

    static random(rows, cols, scale = 0.01) {
        let m = new Matrix(rows, cols);
        for (let i = 0; i < m.data.length; i++) {
            // Box-Muller transform for normal distribution
            let u = 0, v = 0;
            while(u === 0) u = Math.random();
            while(v === 0) v = Math.random();
            let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            m.data[i] = num * scale;
        }
        return m;
    }

    static zero(rows, cols) {
        return new Matrix(rows, cols);
    }

    dot(other) {
        if (this.cols !== other.rows) throw new Error('Incompatible dimensions for dot product');
        let result = new Matrix(this.rows, other.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i * this.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * result.cols + j] = sum;
            }
        }
        return result;
    }

    addBias(bias) {
        let result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i * this.cols + j] = this.data[i * this.cols + j] + bias.data[j];
            }
        }
        return result;
    }

    transpose() {
        let result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j * this.rows + i] = this.data[i * this.cols + j];
            }
        }
        return result;
    }
}

class Layer {
    constructor() {
        this.params = {};
        this.grads = {};
    }
    forward(input) { throw new Error('Not implemented'); }
    backward(error) { throw new Error('Not implemented'); }
}

class Linear extends Layer {
    constructor(inputSize, outputSize, scale = 0.01) {
        super();
        this.params.W = Matrix.random(inputSize, outputSize, scale);
        this.params.b = Matrix.zero(1, outputSize);
        this.input = null;
    }

    forward(input) {
        this.input = input;
        return input.dot(this.params.W).addBias(this.params.b);
    }

    backward(outputError) {
        // dW = input^T * outputError
        let inputT = this.input.transpose();
        this.grads.W = inputT.dot(outputError);

        // db = sum(outputError)
        this.grads.b = new Matrix(1, outputError.cols);
        for (let j = 0; j < outputError.cols; j++) {
            let sum = 0;
            for (let i = 0; i < outputError.rows; i++) {
                sum += outputError.data[i * outputError.cols + j];
            }
            this.grads.b.data[j] = sum;
        }

        // dInput = outputError * W^T
        let WT = this.params.W.transpose();
        return outputError.dot(WT);
    }
}

class ReLU extends Layer {
    constructor() {
        super();
        this.input = null;
    }

    forward(input) {
        this.input = input;
        let result = new Matrix(input.rows, input.cols);
        for (let i = 0; i < input.data.length; i++) {
            result.data[i] = Math.max(0, input.data[i]);
        }
        return result;
    }

    backward(outputError) {
        let result = new Matrix(outputError.rows, outputError.cols);
        for (let i = 0; i < outputError.data.length; i++) {
            result.data[i] = this.input.data[i] > 0 ? outputError.data[i] : 0;
        }
        return result;
    }
}

class Sequential {
    constructor(layers = []) {
        this.layers = layers;
    }

    forward(input) {
        let output = input;
        for (let layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    backward(error) {
        for (let i = this.layers.length - 1; i >= 0; i--) {
            error = this.layers[i].backward(error);
        }
        return error;
    }
}

class CrossEntropyLoss {
    constructor() {
        this.probs = null;
    }

    loss(scores, yTrue) {
        // Softmax
        let rows = scores.rows;
        let cols = scores.cols;
        this.probs = new Matrix(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            let maxVal = -Infinity;
            for (let j = 0; j < cols; j++) {
                if (scores.data[i * cols + j] > maxVal) maxVal = scores.data[i * cols + j];
            }

            let sumExp = 0;
            for (let j = 0; j < cols; j++) {
                let e = Math.exp(scores.data[i * cols + j] - maxVal);
                this.probs.data[i * cols + j] = e;
                sumExp += e;
            }

            for (let j = 0; j < cols; j++) {
                this.probs.data[i * cols + j] /= sumExp;
            }
        }

        // Cross Entropy
        let totalLoss = 0;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (yTrue.data[i * cols + j] === 1) {
                    totalLoss -= Math.log(this.probs.data[i * cols + j] + 1e-10);
                }
            }
        }
        return totalLoss / rows;
    }

    backward(yTrue) {
        let rows = this.probs.rows;
        let cols = this.probs.cols;
        let grad = new Matrix(rows, cols);
        for (let i = 0; i < rows * cols; i++) {
            grad.data[i] = (this.probs.data[i] - yTrue.data[i]) / rows;
        }
        return grad;
    }
}

class SGD {
    constructor(layers, lr = 0.01, reg = 1e-4) {
        this.layers = layers;
        this.lr = lr;
        this.reg = reg;
    }

    step() {
        for (let layer of this.layers) {
            if (layer.params) {
                for (let key in layer.params) {
                    let p = layer.params[key];
                    let g = layer.grads[key];
                    for (let i = 0; i < p.data.length; i++) {
                        let update = g.data[i];
                        if (key === 'W') update += this.reg * p.data[i];
                        p.data[i] -= this.lr * update;
                    }
                }
            }
        }
    }
}

// Export classes for use in app.js
if (typeof module !== 'undefined') {
    module.exports = { Matrix, Linear, ReLU, Sequential, CrossEntropyLoss, SGD };
}
