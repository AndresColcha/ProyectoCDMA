def calculate_total(price: float, tax: float | None) -> float:
    return price + (tax if tax else 0)
