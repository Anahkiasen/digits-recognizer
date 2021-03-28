import * as tf from "@tensorflow/tfjs";
import React, { useRef, useState } from "react";
import CanvasDraw from "react-canvas-draw";
import ReactDOM from "react-dom";
import useModel from "./useModel";

const IMAGE_SIZE = 28;

function App() {
    const model = useModel();
    const [prediction, setPrediction] = useState();

    const onClear = () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        setPrediction(null);
    };

    return (
        <div className="d-flex flex-column align-items-center">
            <CanvasDraw
                className="d-block border rounded mb-5"
                width="200"
                height="200"
                hideGrid
                onChange={(event) => {
                    const { drawing: canvas } = event.canvas;
                    const { drawing: context } = event.ctx;

                    const image = new Image();
                    image.onload = function () {
                        const imageData = context.getImageData(
                            0,
                            0,
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        );

                        const tensor = tf.browser
                            .fromPixels(imageData, 1)
                            .reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);

                        model
                            .predict([tensor])
                            .array()
                            .then(function (scores) {
                                scores = scores[0];
                                console.log(scores);
                                setPrediction(
                                    scores.indexOf(Math.max(...scores))
                                );
                            });
                    };

                    image.src = canvas.toDataURL("image/png");
                }}
            />
            <div className="d-block alert alert-success">{prediction}</div>
            <button onClick={onClear} className="btn btn-info">
                Clear
            </button>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById("app"));
