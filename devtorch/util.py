def cast(data, device, dtype):
    if type(data) == list:
        return [element.to(device).to(dtype) for element in data]
    else:
        return data.to(device).to(dtype)
