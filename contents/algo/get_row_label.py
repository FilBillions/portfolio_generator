def get_row_label(value):
    if value < -8:
        return "<-8%"
    elif -8 <= value < -7:
        return "-8% to -7%"
    elif -7 <= value < -6:
        return "-7% to -6%"
    elif -6 <= value < -5:
        return "-6% to -5%"
    elif -5 <= value < -4:
        return "-5% to -4%"
    elif -4 <= value < -3:
        return "-4% to -3%"
    elif -3 <= value < -2:
        return "-3% to -2%"
    elif -2 <= value < -1:
        return "-2% to -1%"
    elif -1 <= value < 0:
        return "-1% to 0%"
    elif 0 <= value < 1:
        return "0% to 1%"
    elif 1 <= value < 2:
        return "1% to 2%"
    elif 2 <= value < 3:
        return "2% to 3%"
    elif 3 <= value < 4:
        return "3% to 4%"
    elif 4 <= value < 5:
        return "4% to 5%"
    elif 5 <= value < 6:
        return "5% to 6%"
    elif 6 <= value < 7:
        return "6% to 7%"
    elif 7 <= value < 8:
        return "7% to 8%"
    elif value >= 8:
        return ">8%"