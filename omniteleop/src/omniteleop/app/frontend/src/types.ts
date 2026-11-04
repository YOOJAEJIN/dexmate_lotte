// Control states reported by the backend
export type ControlState =
  | 'BOOT'
  | 'DIAGNOSIS'
  | 'ALIGN'
  | 'ACTIVE'
  | 'RECORD'
  | 'PAUSE'
  | 'DEAD'

// One item in the robots array streamed via /ws
export interface StateTopic {
  topic: string
  data_type: string
  data: unknown
}

export interface ObservationMessage {
  timestamp_ns?: number
  joints?: Record<string, number[]>
  velocities?: Record<string, number[]>
}
export interface ActionMessage {
  timestamp_ns?: number
  components?: Record<
    string,
    {
      pos?: number[]
      vel?: number[]
      mode: 'absolute' | 'relative'
    }
  >
  safety_flags?: {
    emergency_stop?: boolean
    exit_requested?: boolean
  }
}
export interface RecordMessage {
  episode_id: string | null
  elapsed_time: number | null
}

// The top-level WebSocket message from /ws (now simplified to robot/joints)
export interface WsStateMessage {
  command?: string
  content?: ObservationMessage | ActionMessage | RecordMessage
}

// Parsed and normalised state used in the UI
export interface RecorderState {
  controlState: ControlState
  isRecording: boolean
  episodeId: string | null
  elapsedTime: number | null
  softwareEstop: boolean
  recordMode: boolean       // backend was started with --record-mode
  teleopRunning: boolean    // teleop subprocess stack is active
  errorState: ErrorStateData | null
  observations: ObservationMessage
  actions: ActionMessage
  activeComponents: string[]
}

export interface ErrorStateData {
  state: string
  message?: string
  jetson_ip?: string
  missing_topics?: string[]
  found_topics?: string[]
  diagnostic_checks?: Record<string, unknown>
  component_errors?: Record<string, unknown>
  out_of_limit?: Record<string, unknown>
}

export interface RobotConfig {
  robot_name: string | null
  robot_config: string | null
  zenoh_config: string | null
  ip: string
  port: number[]
  position?: string[]
}

export interface ConfigResponseSuccess {
  result: 'success'
  content: RobotConfig
}

export interface ConfigResponseFail {
  result: 'fail'
  content: string
}

export type ConfigResponse = ConfigResponseSuccess | ConfigResponseFail

export interface Sensor {
  id: string
  data_type: string
}
