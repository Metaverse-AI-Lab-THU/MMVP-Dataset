from __future__ import absolute_import, division, print_function

import configargparse


def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser

    description = 'Visualization of MMVP GT'

    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='Visualization')

    # config file
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        is_config_file=True,
                        help='config file path')

    # input and output files
    parser.add_argument('--dataset',
                        default='PressureDataset',
                        type=str,
                        help='The name of the dataset that will be used')
    parser.add_argument('--basdir',
                        default='E:/dataset',
                        type=str,
                        help='Base dir')
    parser.add_argument('--sub_ids',
                        default='S01',
                        type=str,
                        help='Subject ids')
    parser.add_argument('--seq_name',
                        default='MoCap_20230422_145333',
                        type=str,
                        help='Sequence name')
    parser.add_argument('--output_dir',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--frame_idx', default=0, type=int)

    parser.add_argument('--model_gender',
                        default='neutral',
                        type=str,
                        help='The gender of item.')

    parser.add_argument('--essential_root',
                        type=str,
                        default='essentials',
                        help='essential files')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict
