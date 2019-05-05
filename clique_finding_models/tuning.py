import copy


def generate_all_configs(variants_dict):
    """Generates all versions of configs, where the leaf lists are substituted with values."""
    for key, values in variants_dict.items():
        if isinstance(values, dict):
            # Transform a dict with unset parameters to a list of setted dicts.
            subconfigs = generate_all_configs(values)
            if len(subconfigs) == 1:
                continue
            values = subconfigs

        if isinstance(values, list):
            # Set one parameter with value from list and recursively set all parameters.
            configs = []
            for value in values:
                setted_dict = copy.deepcopy(variants_dict)
                setted_dict[key] = value
                configs.extend(generate_all_configs(setted_dict))
            return configs
    else:
        # Return dict if all parameters are set (= not lists).
        return [variants_dict]
