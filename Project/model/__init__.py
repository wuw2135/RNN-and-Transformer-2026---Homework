# jiangvicky/MVEO/model/__init__.py
import importlib

def _import_create_model(variant: str):
    """
    根據 variant 載入對應的 create_model 模組
    variant:
        - None 或 "base" 或 ""  →  model.create_model
        - "longtail"            →  model.create_model_longtail
        - 其他字串               →  model.<variant> (例如 model.create_model_dino)
    """
    base_pkg = __package__  # e.g. "jiangvicky.MVEO.model"

    if not variant or variant in ("base", ""):
        module_name = f"{base_pkg}.create_model"
    elif variant == "longtail":
        module_name = f"{base_pkg}.create_model_longtail"
    else:
        # 允許自訂其他 create_model_xxx 模組
        module_name = f"{base_pkg}.{variant}"

    module = importlib.import_module(module_name)
    if not hasattr(module, "create_model"):
        raise ImportError(f"Module '{module_name}' 沒有定義 create_model 函數。")
    return module.create_model


def create_model(opt):
    """
    主入口：根據 opt.create_model 選擇不同版本的 create_model。
    """
    variant = getattr(opt, "create_model", "base")  # 預設 base 版
    create_func = _import_create_model(variant)
    return create_func(opt)
