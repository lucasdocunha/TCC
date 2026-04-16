from src.pipelines.xcpetion import run_xception

if __name__ == "__main__":
    modes_fourier = ["none", "magnitude", "phase", "complex", "concat"]

    for mode_fourier in modes_fourier:
        run_xception(mode_fourier, epochs=20, raw_min=False)
