from datetime import timedelta


def postprocess(start_time, preds: dict):
    results = []

    for h, value in preds.items():
        results.append({
            "index": h - 1,
            "time": (start_time + timedelta(hours=h)).isoformat(),
            "electric": float(value)
        })

    return results
