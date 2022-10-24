import gunpowder as gp


class LogNode(gp.BatchFilter):
    def __init__(self, prefix, log_image_dir=None):
        """Custom gunpowder node for printing data path

        :param str prefix: prefix to print before message (ex: node number)
        :param str log_image_dir: directory to log data to as it travels downstream
                                    by default, is None, and will not log data
        """
        self.prefix = prefix
        self.log_image_dir = log_image_dir

    def prepare(self, request):
        """Prints message during call to upstream request
        :param gp.BatchRequest request: current gp request
        """
        print(f"{self.prefix}\t Upstream provider: {self.upstream_providers[0]}")

    def process(self, batch, request):
        """Prints message during call to downstream request
        :param gp.BatchRequest batch: batch returned by request from memory
        :param gp.BatchRequest request: current gp request
        """
        if self.log_image_dir:
            pass  # TODO implement this using the datalogging utils.
        print(f"{self.prefix}\tBatch going downstream: {batch}")
