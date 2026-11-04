import { useEffect, useRef, useState } from 'react'
import {Slide, toast} from 'react-toastify'
import type {RecorderState, WsStateMessage, RecordMessage, ObservationMessage, ActionMessage} from '../types'
import { httpToWs } from '../lib/utils'

const DEFAULT_STATE: RecorderState = {
  controlState: 'BOOT',
  isRecording: false,
  episodeId: null,
  elapsedTime: null,
  softwareEstop: false,
  recordMode: false,
  teleopRunning: false,
  errorState: null,
  observations: {},
  actions: {},
  activeComponents: [],
}

function parseMessage(msg: WsStateMessage, prevState: RecorderState): RecorderState {
  if (msg.command === 'recording') {
    const content = msg.content as RecordMessage
    return {
      ...prevState,
      episodeId: content?.episode_id ?? null,
      elapsedTime: content?.elapsed_time ?? null,
    }
  }

  if (msg.command === 'save' || msg.command === 'discard') {
    const content = msg.content as RecordMessage
    return {
      ...prevState,
      episodeId: content?.episode_id ?? null,
    }
  }

  if (msg.command === 'observation') {
    const content = msg.content as ObservationMessage
    return {
      ...prevState,
      observations: content,
    }
  }

  if (msg.command === 'action') {
    const content = msg.content as ActionMessage
    return {
      ...prevState,
      actions: content,
    }
  }

  return {
    ...prevState,
  }
}

export function useStateStream(apiUrl: string) {
  const [state, setState] = useState<RecorderState>(DEFAULT_STATE)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let alive = true

    function connect() {
      if (!alive) return
      let ws: WebSocket
      try {
        ws = new WebSocket(`${httpToWs(apiUrl)}/ws`)
      } catch {
        // Invalid URL — retry after delay rather than crashing.
        if (alive) retryRef.current = setTimeout(connect, 2000)
        return
      }
      wsRef.current = ws

      ws.onopen = () => {
        if (!alive) { ws.close(); return }
        setConnected(true)
      }

      ws.onmessage = (ev) => {
        if (!alive) return
        try {
          const msg: WsStateMessage = JSON.parse(ev.data as string)

          if (msg.command === 'save') {
            const content = msg.content as RecordMessage
            toast.info(`episode_id ${content?.episode_id}가 저장되었습니다`)
          } else if (msg.command === 'discard') {
            const content = msg.content as RecordMessage
            toast.warn(`episode_id ${content?.episode_id}가 삭제되었습니다`, {
              transition: Slide,
            })
          }

          setState((prev) => parseMessage(msg, prev))
        } catch {
          // malformed frame – ignore
        }
      }

      ws.onerror = () => {
        setConnected(false)
      }

      ws.onclose = () => {
        setConnected(false)
        if (alive) {
          retryRef.current = setTimeout(connect, 2000)
        }
      }
    }

    connect()

    return () => {
      alive = false
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [apiUrl])

  return { state, connected }
}
