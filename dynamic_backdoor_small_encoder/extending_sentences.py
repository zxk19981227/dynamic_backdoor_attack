import argparse


def main(args: argparse.ArgumentParser.parse_args):
    file_path = args.file_path
    attack_rate = args.attack_rate
    normal_rate = args.noraml_rate
    # attack/normal rate if how many train data is poisoned/normal
    # 1-attack_rate-normal_rate is the negative
