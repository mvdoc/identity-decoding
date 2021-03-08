#!/usr/bin/env python
"""Script to convert logs to events.tsv"""
import argparse
from famfaceangles.bids import make_events


def convert():
    parsed = parse_args()
    with open(parsed.input, 'r') as f:
        lines = f.readlines()
    df = make_events(lines)
    df.to_csv(parsed.output, sep='\t', float_format='%.2f', index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str,
                        help='input file',
                        required=True)
    parser.add_argument('--output', '-o', type=str,
                        help='output file',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    convert()
