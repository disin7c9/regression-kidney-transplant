from utils.reflection import get_class, get_config


class MainProcess:

    def __init__(self, arguments: dict):
        self.args = arguments
        self.phase = self.args.get('phase')
        runner = self.args.get('runner')
        self.runner = self._runner_select(runner)

    def run(self):
        self.runner.run()

    def _runner_select(self, runner: str):
        
        runners_dict = dict(
            scikit_learn2_2_2="runner.scikit_runner2_2_2.Runner",
        )
        return get_class(runners_dict[runner])(self.args)


if __name__ == '__main__':
    task_name = 'kidney_transplant'

    argument_location = '.'.join(['tasks', task_name, 'arguments2_2_2', 'config'])
    args = get_config(argument_location)

    process = MainProcess(args)
    process.run()