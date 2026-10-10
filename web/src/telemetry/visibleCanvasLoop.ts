/**
 * Draw a live chart at a bounded rate, and stop work while it cannot be seen.
 * The default of 30 ms is every second frame on a 60 Hz display, matching the 30 Hz data rate.
 */
export function startVisibleCanvasLoop(
  canvas: HTMLCanvasElement,
  draw: (now: number) => void,
  intervalMs = 30,
): () => void {
  let animationFrameId: number | null = null;
  let visible = typeof IntersectionObserver === 'undefined';
  let disposed = false;
  let lastDrawTime = 0;

  const stop = (): void => {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  };

  const startLoop = (): void => {
    if (disposed || !visible || document.hidden || animationFrameId !== null) {
      return;
    }
    animationFrameId = requestAnimationFrame(loop);
  };

  const loop = (time: number): void => {
    if (time - lastDrawTime >= intervalMs) {
      lastDrawTime = time;
      draw(time);
    }
    if (!disposed && visible && !document.hidden) {
      animationFrameId = requestAnimationFrame(loop);
    }
  };

  const observer =
    typeof IntersectionObserver === 'undefined'
      ? null
      : new IntersectionObserver(([entry]) => {
          visible = entry?.isIntersecting ?? false;
          if (visible) {
            startLoop();
          } else {
            stop();
          }
        });
  observer?.observe(canvas);

  const onVisibilityChange = (): void => {
    if (document.hidden) {
      stop();
    } else {
      startLoop();
    }
  };
  document.addEventListener('visibilitychange', onVisibilityChange);
  startLoop();

  return () => {
    disposed = true;
    stop();
    observer?.disconnect();
    document.removeEventListener('visibilitychange', onVisibilityChange);
  };
}
