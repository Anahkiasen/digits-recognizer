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
                brushColor={"#fff"}
                brushRadius={10}
                canvasHeight={200}
                canvasWidth={200}
                className="img-thumbnail rounded shadow mb-5"
                hideGrid
                lazyRadius={0}
                onChange={onDraw}
                style={{ backgroundColor: "black" }}
            />
            <div className="btn-group mr-2">
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((number) => (
                    <button
                        key={number}
                        type="button"
                        disabled
                        className={`btn btn-${
                            number === prediction ? "primary" : "secondary"
                        }`}
                    >
                        {number}
                    </button>
                ))}
                <button
                    className="btn btn-danger"
                    disabled={prediction === null}
                    onClick={() => {
                        canvas.current.clear();
                        setPrediction(null);
                    }}
                >
                    Clear
                </button>
            </div>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById("app"));
