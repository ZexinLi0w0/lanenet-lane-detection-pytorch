import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Img path")
    parser.add_argument("--model_type", help="Model type", default='ENet')
    parser.add_argument("--model", help="Model path", default='./log/best_model.pth')
    parser.add_argument("--save", help="Directory to save output", default="./test_output")
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--test_num", required=False, type=int, help="Test example number", default=1000)
    parser.add_argument("--log_mode", required=False, type=int, help="Log mode", default=0)
    parser.add_argument("--input", required=False, type=str, help="Input folder path", default='./data/')
    return parser.parse_args()
