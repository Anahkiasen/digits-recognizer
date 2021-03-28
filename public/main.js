import tf from "@tensorflow/tfjs";
import React from "react";
import ReactDOM from "react-dom";
import CanvasDraw from "react-canvas-draw";

ReactDOM.render(
    <CanvasDraw
        className="d-block border rounded mb-5" width="200"
        height="200" hideGrid />,
    document.getElementById("app"),
);
