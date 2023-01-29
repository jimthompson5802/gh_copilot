# Python 3

@DeveloperAPI
@register_encoder("stacked_cnn", IMAGE)
class Stacked2DCNN(ImageEncoder):
    def __init__(
        self,
        height: int,
        width: int,
        conv_layers: Optional[List[Dict]] = None,
        num_conv_layers: Optional[int] = None,
        num_channels: int = None,
        out_channels: int = 32,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int], str] = "valid",
        dilation: Union[int, Tuple[int]] = 1,
        conv_use_bias: bool = True,
        padding_mode: str = "zeros",
        conv_norm: Optional[str] = None,
        conv_norm_params: Optional[Dict[str, Any]] = None,
        conv_activation: str = "relu",
        conv_dropout: int = 0,
        pool_function: str = "max",
        pool_kernel_size: Union[int, Tuple[int]] = 2,
        pool_stride: Union[int, Tuple[int]] = None,
        pool_padding: Union[int, Tuple[int]] = 0,
        pool_dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        fc_layers: Optional[List[Dict]] = None,
        num_fc_layers: Optional[int] = 1,
        output_size: int = 128,
        fc_use_bias: bool = True,
        fc_weights_initializer: str = "xavier_uniform",
        fc_bias_initializer: str = "zeros",
        fc_norm: Optional[str] = None,
        fc_norm_params: Optional[Dict[str, Any]] = None,
        fc_activation: str = "relu",
        fc_dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        # map parameter input feature config names to internal names
        img_height = height
        img_width = width
        first_in_channels = num_channels

        self._input_shape = (first_in_channels, img_height, img_width)

        if first_in_channels is None:
            raise ValueError("first_in_channels must not be None.")

        logger.debug("  Conv2DStack")
        self.conv_stack_2d = Conv2DStack(
            img_height=img_height,
            img_width=img_width,
            layers=conv_layers,
            num_layers=num_conv_layers,
            first_in_channels=first_in_channels,
            default_out_channels=out_channels,
            default_kernel_size=kernel_size,
            default_stride=stride,
            default_padding=padding,
            default_dilation=dilation,
            default_groups=groups,
            default_use_bias=conv_use_bias,
            default_padding_mode=padding_mode,
            default_norm=conv_norm,
            default_norm_params=conv_norm_params,
            default_activation=conv_activation,
            default_dropout=conv_dropout,
            default_pool_function=pool_function,
            default_pool_kernel_size=pool_kernel_size,
            default_pool_stride=pool_stride,
            default_pool_padding=pool_padding,
            default_pool_dilation=pool_dilation,
        )
        out_channels, img_height, img_width = self.conv_stack_2d.output_shape
        first_fc_layer_input_size = out_channels * img_height * img_width

        self.flatten = torch.nn.Flatten()

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_output_size=output_size,
            default_use_bias=fc_use_bias,
            default_weights_initializer=fc_weights_initializer,
            default_bias_initializer=fc_bias_initializer,
            default_norm=fc_norm,
            default_norm_params=fc_norm_params,
            default_activation=fc_activation,
            default_dropout=fc_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The inputs fed into the encoder.
                Shape: [batch x channels x height x width], type torch.uint8
        """

        hidden = self.conv_stack_2d(inputs)
        hidden = self.flatten(hidden)
        outputs = self.fc_stack(hidden)

        return {"encoder_output": outputs}

    @staticmethod
    def get_schema_cls():
        return Stacked2DCNNConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)
