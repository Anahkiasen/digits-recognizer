import * as tf from "@tensorflow/tfjs";
import { useState } from "react";

class L2 {
    static className = "L2";
    constructor(config) {
        return tf.regularizers.l1l2(config);
    }
}

export default function useModel() {
    const [model, setModel] = useState();

    tf.serialization.registerClass(L2);
    tf.loadLayersModel("model.json").then(setModel);

    return model;
}
