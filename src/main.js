import * as tf from "@tensorflow/tfjs";
import React, { useRef, useState } from "react";
import CanvasDraw from "react-canvas-draw";
import ReactDOM from "react-dom";
import useModel from "./useModel";

const IMAGE_SIZE = 28;
const IMAGE_CHANNELS = 1;

function App() {
    const model = useModel();
    const [prediction, setPrediction] = useState();
    const [context, setContext] = useState();
    const [canvas, setCanvas] = useState();

    const onClear = () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        setPrediction(null);
    };

    const onDraw = (event) => {
        const { drawing: canvas } = event.canvas;
        const { drawing: context } = event.ctx;

        setCanvas(canvas);
        setContext(context);

        const tensor = tf.image.resizeBilinear(
            tf.browser.fromPixels(canvas, IMAGE_CHANNELS),
            [IMAGE_SIZE, IMAGE_SIZE]
        );

        model
            .predict(
                tensor.reshape([1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
            )
            .array()
            .then(function ([scores]) {
                setPrediction(scores.indexOf(Math.max(...scores)));
            });
    };

    return (
        <div className="d-flex flex-column align-items-center">
            <CanvasDraw
                className="d-block border rounded mb-5"
                brushColor={"#fff"}
                style={{ backgroundColor: "black" }}
                canvasWidth={100}
                canvasHeight={100}
                lazyRadius={0}
                brushRadius={5}
                hideGrid
                onChange={onDraw}
            />
            {prediction !== null && (
                <>
                    <div className="d-block alert alert-success">
                        {prediction}
                    </div>
                    <button onClick={onClear} className="btn btn-info">
                        Clear
                    </button>
                </>
            )}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById("app"));
