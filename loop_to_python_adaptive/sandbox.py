import loop_to_python_adaptive.api as api  # replace with your real import

print("has get_glucose_effect_velocity:", hasattr(api, "get_glucose_effect_velocity"))
print("callable:", callable(getattr(api, "get_glucose_effect_velocity", None)))
print("module file:", api.__file__)