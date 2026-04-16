from src.pipelines.xcpetion import run_xception
from src.pipelines.vit import run_vit

if __name__ == "__main__":
    # modes_fourier = ["none", "magnitude", "phase", "complex", "concat"]

    # for mode_fourier in modes_fourier:
    #     run_xception(mode_fourier, epochs=2, raw_min=True)

    run_vit()
