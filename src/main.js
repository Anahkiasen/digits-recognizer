import * as tf from "@tensorflow/tfjs";
import React, { useRef, useState } from "react";
import CanvasDraw from "react-canvas-draw";
import ReactDOM from "react-dom";
import useModel from "./useModel";

const IMAGE_SIZE = 28;
const IMAGE_CHANNELS = 1;

function App() {
    const model = useModel();
    const canvas = useRef();
    const [prediction, setPrediction] = useState(null);

    const onDraw = (event) => {
        const tensor = tf.image.resizeBilinear(
            tf.browser.fromPixels(event.canvas.drawing, IMAGE_CHANNELS),
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
            <h1 className={"mb-5"}>Shitty Digits Recognizer</h1>
            <CanvasDraw
                ref={canvas}
                className="d-block border rounded shadow mb-5"
                brushColor={"#fff"}
                style={{ backgroundColor: "black" }}
                canvasWidth={200}
                canvasHeight={200}
                lazyRadius={0}
                brushRadius={10}
                hideGrid
                onChange={onDraw}
            />
            {prediction !== null && (
                <>
                    <div className="alert alert-success">
                        This is a <strong>{prediction}</strong>
                    </div>
                    <button
                        className="btn btn-info"
                        onClick={() => {
                            canvas.current.clear();
                            setPrediction(null);
                        }}
                    >
                        Clear
                    </button>
                </>
            )}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById("app"));
