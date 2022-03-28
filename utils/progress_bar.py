class ProgressBar:
    _PROGRESS_BAR_LENGTH = 40
    _UPDATE_FREQ = 10

    def __init__(self, total: int, update_freq: int = _UPDATE_FREQ, num_bars: int = _PROGRESS_BAR_LENGTH):
        if total < 1:
            raise ValueError("Total value must be a positive number!")

        if num_bars < 1:
            raise ValueError("Number of bars must be a positive number!")

        if update_freq < 1:
            raise ValueError("Update frequency must be a positive number!")

        self._total = total
        self._num_bars = num_bars
        self._update_freq = update_freq
        self.__enter__()

    def __enter__(self):
        self._current = -1
        self.update()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._print_progress(self._get_progress(), "\n")
        self._current = -1

    def end(self) -> None:
        '''End the current progress bar.'''
        self.__exit__()

    def update(self) -> None:
        '''Updates the count of currently processed items and prints the updated progress bar.'''
        self._current += 1
        if self._current % self._update_freq != 0:
            return
        
        self._print_progress(self._get_progress())

    def _get_progress(self) -> int:
        return min(int(self._current / self._total * self._num_bars), self._num_bars)

    def _print_progress(self, progress: int, end: str = "\r") -> None:
        print(f"Currently processed: [{'=' * progress}{' ' * (self._num_bars - progress)}] {self._current}/{self._total} files - [{self._current/self._total * 100:.0f}%]", end=end, flush=True)
    