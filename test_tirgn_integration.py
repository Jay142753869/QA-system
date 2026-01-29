from config import Config
from core.tirgn_wrapper import TiRGNWrapper


def main():
    cfg = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    wrapper = TiRGNWrapper(cfg)
    results = wrapper.predict_tail("招商银行", "监事会提名委员会委员", "20250102", top_k=5)
    assert isinstance(results, list)
    assert all(isinstance(x, dict) for x in results)
    print("TiRGN OK:", results[:5])


if __name__ == "__main__":
    main()

