from mothernet import TabPFNClassifier
import os
from pathlib import Path
import torch



class EquiTabPFNClassifier(TabPFNClassifier):
    def __init__(
        self,
        epoch: int = 1200,
        N_ensemble_configurations: int = 1,
        batch_size_inference: int = 32,
        device: str = "cpu",
        max_class: int = 20,
        compile: bool = True,
    ):
        if "EQUIPFN_CHECKPOINT" in os.environ:
            checkpoint_path = Path(os.getenv("EQUIPFN_CHECKPOINT"))
        else:
            checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints/"
        model_path = checkpoint_path / "equipfn_w_mask/epoch_1200"
        print(f"model_path: {model_path}")
        if not model_path.exists():
            print(f"{model_path} does not exists, try to download it.")
            download_model("equipfn_w_mask/epoch_1200", checkpoint_path)
            assert model_path.exists()

        model_string = "train"
        epoch = -1
        model_key = model_string + "|" + str(device) + "|" + str(epoch)
        model, args = load_model(model_path, device, verbose=False)

        max_num_classes = max_class
        args["prior"]["classification"]["max_num_classes"] = max_num_classes
        args["max_num_classes"] = max_num_classes
        from equitabpfn.utils import OneHot

        model.y_encoder.one_hot = OneHot(max_num_classes)
        model.to(device)
        if compile:
            model = torch.compile(model)
        classif = TabPFNClassifier(
            device=device,
            base_path="",
            model_string=model_string,
            N_ensemble_configurations=N_ensemble_configurations,
            feature_shift_decoder=True,
            multiclass_decoder="",
            no_preprocess_mode=False,
            batch_size_inference=batch_size_inference,
            epoch=epoch,
        )

        classif.models_in_memory[model_key] = (model, args, "")
        classif.max_num_classes = max_num_classes

        self.classif = classif
        self.classif.c = args

    def fit(self, X, y, overwrite_warning=False):
        self.classif.fit(X, y, overwrite_warning=overwrite_warning)
        return self.classif

    def predict(self, X, **kwargs):
        return self.classif.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.classif.predict_proba(X, **kwargs)


if __name__ == "__main__":
    import numpy as np
    from timeblock import Timeblock

    N = 1000
    m = 10
    d = 100

    def make_model(use_equi: bool, device: str, batch_size_inference: int):
        if use_equi:
            return EquiTabPFNClassifier(
                batch_size_inference=batch_size_inference, compile=True, device=device
            )
        else:
            from mothernet.prediction.tabpfn import TabPFNClassifier

            classifier = TabPFNClassifier(
                device=device,
                N_ensemble_configurations=1,
                batch_size_inference=batch_size_inference,
                # epoch=42,
            )
            return classifier

    print("TabPFN runtime")
    devices = [
        "cuda",
        "cpu",
    ]
    for device in devices:

        print(device)
        with Timeblock("loading model"):
            classif = TabPFNClassifier(
                device=device,
                N_ensemble_configurations=1,
                batch_size_inference=N,
                # epoch=42,
            )

        # dummy code to compile the model
        X = np.random.rand(10, 2)
        y = np.random.randint(low=0, high=3, size=10)
        classif.fit(X, y)
        # classif.model = torch.compile(classif.model)

        for k in range(3):
            with Timeblock(f"fit & predict {k}"):
                X = np.random.rand(N, d)
                y = np.random.randint(low=0, high=m, size=N)
                classif.fit(X, y)
                print(classif.predict(X).shape)

    print("Equitabpfn runtime")
    for device in devices:
        print(device)
        with Timeblock("loading model"):
            classif = EquiTabPFNClassifier(
                batch_size_inference=N,
                compile=True,
                device=device,
                N_ensemble_configurations=1,
            )

        for k in range(3):
            with Timeblock(f"fit & predict {k}"):
                X = np.random.rand(N, d)
                y = np.random.randint(low=0, high=m, size=N)
                classif.fit(X, y)
                print(classif.predict(X).shape)