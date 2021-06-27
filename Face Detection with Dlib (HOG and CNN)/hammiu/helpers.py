def convert_anh_trim_bb(image, rect):
    # trích xuất tọa độ đầu cuối của bounding box
    # vẽ hình ra là thấy ngay
    x_min = rect.left()
    y_min = rect.top()
    x_max = rect.right()
    y_max = rect.bottom()

    # đảm bảo tọa độ bounding box nằm trong giới hạn của ảnh
    x_min = max(0, x_min)   # TH x_min nhỏ hơn 0 phải cắt
    y_min = max(0, y_min)   # TH y_min nhỏ hơn 0 phải cắt
    x_max = min(x_max, image.shape[1])      # TH x_max lớn hơn width phải cắt
    y_max = min(y_max, image.shape[0])      # TH y_max lớn hơn height phải cắt

    # tính width, height của bounding box
    w = x_max - x_min
    h = y_max - y_min 

    return (x_min, y_min, w, h)     # tọa độ góc trên bên trái, width, height

    
