import gunpowder as gp
import numpy as np


class AugmentationNodeBuilder:
    """
    Optimally builds augmentations nodes given some augmentation config to
    maintain sequential compatibility and computational efficiency.
    """

    def __init__(
        self,
        augmentation_config,
        spatial_dims=2,
        noise_key=None,
        defect_key=None,
        intensities_key=None,
        blur_key=None,
    ):
        self.config = augmentation_config

        # augmentations with default parameters
        self.elastic_aug_params = {
            "control_point_spacing": (1, 1, 1),
            "jitter_sigma": (0, 0, 0),
            "spatial_dims": spatial_dims,
            "use_fast_points_transform": False,
            "subsample": 1,
        }
        self.simple_aug_params = {}
        self.shear_aug_params = {}
        self.intensity_aug_params = {"intensity_array_key": intensities_key}
        self.noise_aug_params = {"noise_array_key": noise_key}
        self.blur_aug_params = {"blur_array_key": blur_key}
        self.defect_aug_params = {"intensities_defect_key": defect_key}

        self.node_list = []

    def build_nodes(self):
        """
        Build list of augmentation nodes in compatible sequential order going downstream

        Strict ordering levels (going downstream):
        1 -> shear, simple (mirror & transpose)
        2 -> elastic (zoom, rotation)
        3 -> blur
        4 -> intensity, noise, defect (contrast & artifacts)
        """
        ordering = [
            {"transpose", "mirror"},
            {"rotate", "zoom"},
            {"blur"},
            {"intensity_jitter", "noise", "contrast_shift", "defect"},
        ]
        for subset in ordering:
            for name in subset:
                if name in self.config:
                    aug_node = self.init_aug_node(name, self.config[name])
                    self.node_list.append(aug_node)

    def get_nodes(self):
        """
        Getter for nodes.

        :return list node_list: list of initalized augmentation nodes
        """
        assert (
            len(self.node_list) > 0
        ), "Augmentation nodes not initiated or unspecified. "
        "Try .build_nodes() or check your config"

        return self.node_list

    def init_aug_node(self, aug_name, parameters):
        """
        Acts as a general initatialization method, which takes a augmentation name and
        parameters and initializes and returns a gunpowder node corresponding to that
        augmentation

        :param str aug_name: name of augmentation
        :param dict parameters: dict of parameter names and values for augmentation

        :return gp.BatchFilter aug_node: single gunpowder node for augmentation
        """

        # TODO Will need a discussion about what augmentations to support
        if aug_name in {"transpose", "mirror"}:
            self.simple_aug_params.update(parameters)
        elif aug_name in {"rotate", "zoom"}:
            self.elastic_aug_params.update(parameters)
        elif aug_name == "shear":
            self.shear_aug_params.update(parameters)
        elif aug_name == "intensity_jitter":
            self.intensity_aug_params.update(parameters)
        elif aug_name == "noise":
            assert self.noise_aug_params["array"] != None, "No noise key specified."
            self.noise_aug_params.update(parameters)
        elif aug_name == "blur":
            self.blur_aug_params.update(parameters)
        elif aug_name == "contrast_shift":  # TODO explore
            pass
        elif aug_name == "defect":  # TODO explore
            pass

    def build_elastic_augment_node(self, parameters):
        """
        passes parameters to elastic augmentation node and returns initialized node

        :param dict parameters: elastic augmentation node parameters
        :return gp.BatchFilter: elastic augmentation node
        """

        rotation_interval = (0, 0)
        if "rotation_interval" in self.elastic_aug_params:
            rotation_interval = tuple(self.elastic_aug_params["rotation_interval"])

        scale_interval = (0, 0)
        if "scale_interval" in self.elastic_aug_params:
            scale_interval = tuple(self.elastic_aug_params["scale_interval"])

        elastic_aug = gp.ElasticAugment(
            rotation_interval=rotation_interval,
            scale_interval=scale_interval,
            control_point_spacing=self.elastic_aug_params["control_point_spacing"],
            jitter_sigma=self.elastic_aug_params["jitter_sigma"],
            spatial_dims=self.elastic_aug_params["spatial_dims"],
            use_fast_points_transform=self.elastic_aug_params[
                "use_fast_points_transform"
            ],
            subsample=self.elastic_aug_params["subsample"],
        )
        return elastic_aug

    def build_simple_augment_node(self, parameters):
        """
        passes parameters to simple augmentation node and returns initialized node

        :param dict parameters: simple augmentation node parameters
        :return gp.BatchFilter: simple augmentation node
        """

        transpose_only = None
        transpose_probs = None
        if "transpose" in self.simple_aug_params:
            transpose_only = tuple(self.simple_aug_params["transpose_only"])
        else:
            transpose_probs = (0) * 10

        mirror_only = None
        mirror_probs = None
        if "mirror" in self.simple_aug_params:
            mirror_only = self.simple_aug_params["mirror_only"]
        else:
            mirror_probs = (0) * 10  # assuming 10 > all reasonable voxel dimensions

        simple_aug = gp.SimpleAugment(
            transpose_only=transpose_only,
            mirror_only=mirror_only,
            transpose_probs=transpose_probs,
            mirror_probs=mirror_probs,
        )
        return simple_aug

    def build_shear_augment_node(self, parameters):
        """
        passes parameters to shear augmentation node and returns initialized node

        :param dict parameters: shear augmentation node parameters
        :return gp.BatchFilter: shear augmentation node
        """
        # TODO implement
        pass

    def build_blur_augment_node(self, parameters):
        """
        passes parameters to blur augmentation node and returns initialized node

        :param dict parameters: blur augmentation node parameters
        :return gp.BatchFilter: blur augmentation node
        """
        # TODO implement
        pass

    def build_intensity_augment_node(self, parameters):
        """
        passes parameters to intensity augmentation node and returns initialized node

        :param dict parameters: intensity augmentation node parameters
        :return gp.BatchFilter: intensity augmentation node
        """
        # TODO implement
        pass

    def build_noise_augment_node(self, parameters):
        """
        passes parameters to noise augmentation node and returns initialized node

        :param dict parameters: noise augmentation node parameters
        :return gp.BatchFilter: noise augmentation node
        """
        # TODO implement
        pass
