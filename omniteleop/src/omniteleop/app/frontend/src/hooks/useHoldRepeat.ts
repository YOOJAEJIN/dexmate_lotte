import React, { useCallback, useEffect, useRef } from 'react'

/**
 * Press-and-hold repeat: run an action immediately on start, then repeat it
 * every intervalMs until stop (pointer up / cancel / unmount).
 *
 * Used by the arm/head/torso jog buttons: unlike the chassis (latched
 * velocity, needs an explicit zero on release), these send relative position
 * deltas that are consumed once - the robot settles at the last target and
 * stops on its own, so releasing simply stops the repeats.
 *
 * The action is captured at press time so e.g. changing the joint selection
 * mid-hold does not affect the running repeat.
 */
export function useHoldRepeat(intervalMs = 100) {
  const timerRef = useRef<number | null>(null)
  const actionRef = useRef<(() => void) | null>(null)

  const stop = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
    actionRef.current = null
  }, [])

  const start = useCallback(
    (action: () => void) => {
      stop()
      actionRef.current = action
      action()
      timerRef.current = window.setInterval(() => {
        actionRef.current?.()
      }, intervalMs)
    },
    [stop, intervalMs]
  )

  // Stop repeating if the component unmounts mid-hold
  useEffect(() => stop, [stop])

  return { start, stop }
}

/**
 * Wire a button to useHoldRepeat: pointerdown captures the pointer (so
 * pointerup is delivered even if the pointer leaves the button) and starts
 * repeating; pointerup/cancel stops.
 */
export function holdPointerHandlers(
  hold: { start: (action: () => void) => void; stop: () => void },
  action: () => void,
  disabled?: boolean
) {
  return {
    onPointerDown: (e: React.PointerEvent<HTMLButtonElement>) => {
      if (disabled) return
      e.currentTarget.setPointerCapture(e.pointerId)
      hold.start(action)
    },
    onPointerUp: () => hold.stop(),
    onPointerCancel: () => hold.stop(),
  }
}
