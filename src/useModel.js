import * as tf from "@tensorflow/tfjs";
import { useEffect, useState } from "react";

class L2 {
    static className = "L2";

    constructor(config) {
        return tf.regularizers.l1l2(config);
    }
}

export default function useModel() {
    const [model, setModel] = useState();

    useEffect(() => {
        if (model) {
            return;
        }

        tf.serialization.registerClass(L2);
        tf.loadLayersModel("model.json").then(setModel);
    }, [model]);

    return model;
}
