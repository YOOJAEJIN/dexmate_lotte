import {useState, useEffect} from 'react'
import {toast, ToastContainer, Zoom} from 'react-toastify';
import lgcnsLogo from './assets/lgcns-logo.svg'
import { Settings, Moon, Sun } from 'lucide-react'
import { HashLoader } from "react-spinners";
import { useStateStream } from './hooks/useStateStream'
import { TeleopControls } from './components/TeleopControls'
// import { StatusPanel } from './components/StatusPanel'
import { RobotStatus } from './components/RobotStatus'
import { ObservationPanel } from './components/ObservationPanel.tsx'
import { ActionPanel } from './components/ActionPanel'
import { ArmControl } from './components/ArmControl'
import { TorsoControl } from './components/TorsoControl'
import { ChassisControl } from './components/ChassisControl'
import { PredefinedPoseControl } from './components/PredefinedPoseControl'
import {ReachControl} from "./components/ReachControl.tsx";
import { DebugPanel } from './components/DebugPanel'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { type TeleopLaunchConfig } from './components/TeleopControls'
import { type RobotConfig, type ConfigResponse } from './types'

const DEFAULT_API = `${window.location.protocol}//${window.location.hostname}:5006`

type SidePanel = 'settings' | 'debug'

function loadPersisted(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}
function persist(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* noop */ }
}

export default function App() {
  const [apiUrl, setApiUrl]     = useState(() => loadPersisted('apiUrl', DEFAULT_API))
  const [apiInput, setApiInput] = useState(apiUrl)
  const [isActive, setIsActive] = useState(true)
  const [configError, setConfigError] = useState<string | null>(null)
  const [loading, setLoading]   = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [sidePanel, setSidePanel] = useState<SidePanel | null>(null)
  const [dark, setDark] = useState(() => loadPersisted('theme', 'light') === 'dark')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    persist('theme', dark ? 'dark' : 'light')
  }, [dark])

  const [showDiagnosis, setShowDiagnosis] = useState(false)

  const { state, connected } = useStateStream(apiUrl)
  const [config, setConfig] = useState<RobotConfig | null>(null)
  const [isRecording, setIsRecording] = useState<boolean>(false)
  const [teleopRunning, setTeleopRunning] = useState<boolean>(false)

  const {
    // controlState,
    episodeId,
    elapsedTime,
    softwareEstop,
    // recordMode,
    errorState,
    // activeComponents,
  } = state

  // Auto-open diagnosis modal when state transitions to DIAGNOSIS
  // useEffect(() => {
  //   if (controlState === 'DIAGNOSIS') setShowDiagnosis(true)
  //   else setShowDiagnosis(false)
  // }, [controlState])


  function applyApiUrl(url: string) {
    let trimmed = url.trim().replace(/\/$/, '')
    // If no protocol is given, assume http:// so WebSocket conversion works correctly.
    if (trimmed && !/^https?:\/\//i.test(trimmed)) {
      trimmed = `http://${trimmed}`
    }
    // If no port is present, append the default backend port.
    try {
      const parsed = new URL(trimmed)
      if (!parsed.port) trimmed = `${parsed.protocol}//${parsed.hostname}:5006${parsed.pathname === '/' ? '' : parsed.pathname}`
    } catch { /* invalid URL – pass through and let the connection fail gracefully */ }
    setApiUrl(trimmed)
    setApiInput(trimmed)
    persist('apiUrl', trimmed)
  }

  async function api(method: string, path: string, body?: unknown) {
    setLoading(true)
    setActionError(null)
    try {
      const res = await fetch(`${apiUrl}${path}`, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
      })
      if (!res.ok) {
        const text = await res.text()
        const errorMsg = `${res.status}: ${text}`
        toast.error(text)
        throw new Error(errorMsg)
      }

      // Check for custom error results in JSON response
      const data = await res.json()
      return data
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const fetchState = async () => {
    const data = await api('GET', '/state')
    if (data) {
      setIsRecording(data.is_recording)
      setTeleopRunning(data.teleop_running)
    }
  }
  const handleTeleopStart  = async (cfg: TeleopLaunchConfig) => {
    await api('POST', '/teleop/start', cfg)
    await fetchState()
  }
  const handleTeleopStop   = async () => {
    await api('POST', '/teleop/stop')
    await fetchState()
  }
  const handleReset       = async () => {
    await api('POST', '/teleop/reset')
  }
  const handleResetAll    = async () => {
    await api('POST', '/teleop/resetall')
  }
  const handleRecordStart  = async () => {
    await api('POST', '/record/start', { metadata: {} })
    await fetchState()
  }
  const handleRecordSave   = async () => {
    await api('POST', '/record/stop',  { is_success: true })
    await fetchState()
  }
  const handleRecordDiscard= async () => {
    await api('POST', '/record/stop',  { is_success: false })
    await fetchState()
  }
  const handleEstop        = async () => await api('POST', '/robots/estop')
  const handleClearEstop   = async () => await api('DELETE', '/robots/estop')

  const fetchConfig = async () => {
    const data = await api('GET', '/config') as ConfigResponse
    if (data) {
      if (data.result === 'fail') {
        setIsActive(false)
        setConfigError(data.content)
      } else {
        setIsActive(true)
        setConfigError(null)
        setConfig(data.content)
      }
    }
  }

  useEffect(() => {
    const init = async () => {
      await fetchConfig()
      await fetchState()
    }
    init()
  }, [apiUrl])

  return (
    <div className="min-h-screen bg-muted dark:bg-background">

      {/* ---- Top bar ---- */}
      <header className="sticky top-0 z-20 bg-white dark:bg-muted border-b border-border">
        <div className="max-w-screen-xl mx-auto px-6 h-12 flex items-center gap-3 relative">
          {/* Logo + title */}
          <img
            src={lgcnsLogo}
            alt="lgcns"
            className="h-5 w-auto shrink-0"
          />
          <div className="flex flex-col leading-none">
            <span className="font-semibold text-sm tracking-tight">Future Robotics LAB</span>
          </div>

          {/* robot_name */}
          {config?.robot_name && (
            <div className="absolute left-1/2 -translate-x-1/2 flex items-center px-4 py-1 bg-sky-50 dark:bg-sky-950/40 rounded-full border border-sky-100 dark:border-sky-900/50 shadow-sm">
              <span className="text-lg font-bold text-sky-900 dark:text-sky-100 uppercase tracking-tight">{config.robot_name}</span>
            </div>
          )}

          <div className="ml-auto flex items-center gap-3">
            <span
              title={connected ? 'Backend connected' : 'Backend disconnected'}
              className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-400'}`}
            />
            {/* Dark mode toggle */}
            <button
              onClick={() => setDark(d => !d)}
              className="p-1.5 rounded hover:bg-muted transition-colors"
              title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {dark ? <Sun className="h-4 w-4 text-muted-foreground" /> : <Moon className="h-4 w-4 text-muted-foreground" />}
            </button>
            <div className="flex items-center rounded-md border border-border overflow-hidden">
              <button
                onClick={() => setSidePanel(p => p === 'settings' ? null : 'settings')}
                className={`p-1.5 transition-colors ${sidePanel === 'settings' ? 'bg-muted' : 'hover:bg-muted'}`}
                aria-label="Settings"
                title="Settings"
              >
                <Settings className="h-4 w-4 text-muted-foreground" />
              </button>
              <button
                onClick={() => setSidePanel(p => p === 'debug' ? null : 'debug')}
                className={`px-2.5 h-7 text-xs font-mono transition-colors border-l border-border ${sidePanel === 'debug' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:bg-muted'}`}
                aria-label="Debug"
                title="Debug logs"
              >
                debug
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-6 pt-2 space-y-4">

        {/* ---- Settings / Debug panel ---- */}
        {sidePanel === 'settings' && (
          <Card className="border-dashed">
            <CardHeader><CardTitle>Settings</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-1 max-w-sm">
                <label className="text-xs font-medium text-muted-foreground">Backend URL</label>
                <div className="flex gap-2">
                  <input
                    className="flex-1 h-8 px-2.5 rounded-md border border-input bg-background text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    value={apiInput}
                    onChange={e => setApiInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && applyApiUrl(apiInput)}
                    placeholder={`${window.location.protocol}//${window.location.hostname}:5006`}
                  />
                  <button
                    onClick={() => applyApiUrl(apiInput)}
                    className="px-3 h-8 rounded-md bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90"
                  >
                    Apply
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        {/* ---- Error banner ---- */}
        {actionError && (
          <div className="flex items-start gap-2 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            <span className="font-medium">Error:</span>
            <span className="flex-1">{actionError}</span>
            <button onClick={() => setActionError(null)} className="ml-2 text-red-400 hover:text-red-600">✕</button>
          </div>
        )}

        {/* ---- Main grid: WebRTC 9 : Controls 21 : Status 12 ----
            (이전 3:6:5 비율에서 오른쪽 Status 컬럼만 약 80%로 줄이고 남는 폭을 가운데에 배분) */}
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,9fr)_minmax(0,21fr)_minmax(0,12fr)] gap-6 items-start">

          {/* Left Column: WebRTC (9/42) */}
          <div className="flex flex-col gap-4">
            <Card className="overflow-hidden border-border shadow-md">
              <CardHeader className="py-2.5 px-4 bg-gray-700 text-primary-foreground flex flex-row items-center justify-between space-y-0">
                <CardTitle className="text-xs font-bold tracking-tight text-primary-foreground normal-case">WebRTC Streams</CardTitle>
                <a
                  href={`https://${config?.ip || window.location.hostname}:${config?.port?.[0] || 60001}`}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[10px] font-bold text-sky-200 hover:text-white transition-colors flex items-center gap-1"
                >
                  Authorize <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
                </a>
              </CardHeader>
              {/* 스트림이 많아도 페이지가 길어지지 않도록 카드 안에서만 세로 스크롤 (뷰포트 높이 기준) */}
              <CardContent className="p-3 bg-muted/30 dark:bg-zinc-900/50 flex flex-col gap-3 overflow-y-auto max-h-[calc(100vh-7rem)]">
                {(config?.port || []).map((port, idx) => (
                  <div key={port} className="flex flex-col gap-1.5">
                    <h6 className="text-[10px] font-bold text-muted-foreground/80 uppercase tracking-widest pl-1">
                      {config?.position?.[idx]?.replace(/_/g, ' ') || `Camera ${idx + 1}`}
                    </h6>
                    <div className="relative aspect-[4/3] w-full bg-black rounded-lg overflow-hidden border border-border shadow-inner group">
                      <iframe
                        src={`https://${config?.ip || window.location.hostname}:${port}`}
                        className="w-full h-full border-0 block overflow-hidden"
                        title={`camera-${idx}`}
                        scrolling="no"
                      />
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Middle Column: Controls (21/42) */}
          <div className="flex flex-col gap-4">
            {/* Teleop Controls */}
            <TeleopControls
              teleopRunning={teleopRunning}
              isRecording={isRecording}
              episodeId={episodeId}
              elapsedTime={elapsedTime}
              softwareEstop={softwareEstop}
              loading={loading}
              onTeleopStart={handleTeleopStart}
              onTeleopStop={handleTeleopStop}
              onRecordStart={handleRecordStart}
              onRecordSave={handleRecordSave}
              onRecordDiscard={handleRecordDiscard}
              onReset={handleReset}
              onResetAll={handleResetAll}
              onEstop={handleEstop}
              onClearEstop={handleClearEstop}
            />
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <ArmControl apiUrl={apiUrl} disabled={!teleopRunning} />
              <TorsoControl apiUrl={apiUrl} disabled={!teleopRunning} />
              <ChassisControl apiUrl={apiUrl} disabled={!teleopRunning} />
            </div>
            <PredefinedPoseControl apiUrl={apiUrl} disabled={!teleopRunning} />
            <ReachControl apiUrl={apiUrl} disabled={!teleopRunning} />
          </div>

          {/* Right Column: Robot Status / Observation / Action (12/42) */}
          <div className="grid grid-cols-1 gap-4 items-start">
            <RobotStatus observations={state.observations} />

            <ObservationPanel observations={teleopRunning ? state.observations : {}} accentColor="bg-emerald-800" />
            <ActionPanel
              title="Action Command"
              actions={teleopRunning ? state.actions : {}}
              accentColor="bg-emerald-800"
            />
          </div>

        </div>

        {/* ---- Debug panel (bottom) ---- */}
        {sidePanel === 'debug' && (
          <Card className="border-dashed bg-zinc-950 text-zinc-100">
            <CardHeader className="pb-2">
              <CardTitle className="text-zinc-100 font-mono text-sm">Debug — Process Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <DebugPanel apiUrl={apiUrl} />
            </CardContent>
          </Card>
        )}

      </main>

      {/* ---- Diagnosis modal ---- */}
      {showDiagnosis && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 px-5 py-4 border-b border-border bg-amber-50 dark:bg-amber-950/40">
              <svg className="h-5 w-5 text-amber-600 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              </svg>
              <span className="font-bold text-amber-800 dark:text-amber-200 text-sm tracking-wide">DIAGNOSIS — Teleop cannot start</span>
              <button
                onClick={() => setShowDiagnosis(false)}
                className="ml-auto text-amber-500 hover:text-amber-700 dark:hover:text-amber-300 text-lg leading-none"
              >✕</button>
            </div>
            {/* Body */}
            <div className="px-5 py-4 space-y-4 text-sm max-h-[60vh] overflow-y-auto">
              {errorState?.message && (
                <p className="text-foreground font-medium">{errorState.message}</p>
              )}

              {/* Diagnostic checks: pass/fail table */}
              {errorState?.diagnostic_checks && Object.keys(errorState.diagnostic_checks).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Diagnostic checks</p>
                  <div className="rounded-md border border-border overflow-hidden">
                    {Object.entries(errorState.diagnostic_checks).map(([check, passed]) => (
                      <div key={check} className="flex items-center justify-between px-3 py-1.5 odd:bg-muted/40 text-xs">
                        <span className="font-mono text-foreground">{check.replace(/_/g, ' ')}</span>
                        {passed ? (
                          <span className="text-emerald-600 dark:text-emerald-400 font-semibold">✓ pass</span>
                        ) : (
                          <span className="text-red-600 dark:text-red-400 font-semibold">✗ fail</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Component errors */}
              {errorState?.component_errors && Object.keys(errorState.component_errors).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Component errors</p>
                  <div className="rounded-md border border-border overflow-hidden">
                    {Object.entries(errorState.component_errors).map(([comp, detail]) => (
                      <div key={comp} className="px-3 py-1.5 odd:bg-muted/40 text-xs">
                        <span className="font-mono font-semibold text-red-600 dark:text-red-400">{comp}</span>
                        <span className="ml-2 text-muted-foreground">{JSON.stringify(detail)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Missing topics (BOOT state) */}
              {errorState?.missing_topics && errorState.missing_topics.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Missing topics</p>
                  <ul className="space-y-0.5">
                    {errorState.missing_topics.map(t => (
                      <li key={t} className="font-mono text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 px-2 py-0.5 rounded">{t}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Out-of-limit joints (ALIGN state) */}
              {errorState?.out_of_limit && (() => {
                const { message: _m, ...arms } = errorState.out_of_limit as Record<string, unknown>
                return Object.keys(arms).length > 0 ? (
                  <div>
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Out of limit joints</p>
                    {Object.entries(arms).map(([arm, joints]) => (
                      <div key={arm} className="mb-1">
                        <p className="text-xs font-semibold capitalize mb-0.5">{arm.replace(/_/g, ' ')}</p>
                        <ul className="space-y-0.5">
                          {(joints as string[]).map((j, i) => (
                            <li key={i} className="font-mono text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 px-2 py-0.5 rounded">{j}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : null
              })()}

              {!errorState && (
                <p className="text-muted-foreground italic">No diagnostic details available.</p>
              )}
            </div>
            {/* Footer */}
            <div className="px-5 py-3 border-t border-border flex justify-end">
              <button
                onClick={() => setShowDiagnosis(false)}
                className="px-4 py-1.5 rounded-md bg-muted hover:bg-muted/80 text-sm font-medium transition-colors"
              >Dismiss</button>
            </div>
          </div>
        </div>
      )}

      {loading && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm pointer-events-auto">
            <HashLoader
                color="#038d99"
                loading={loading}
                size={100}
                // cssOverride={{
                //   borderWidth: "8px",
                //   borderColor: "#038d99",
                // }}
                aria-label="Loading Spinner"
                data-testid="loader"
            />
          </div>
      )}

      { !isActive && (
           <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-md pointer-events-auto">
             <div className="bg-white dark:bg-zinc-900 p-8 rounded-xl shadow-2xl border border-red-500/50 max-w-md w-full text-center space-y-4 mx-4">
               <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-100 dark:bg-red-950/40 mb-2">
                 <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                 </svg>
               </div>
               <h2 className="text-2xl font-bold text-foreground tracking-tight">Configuration Error</h2>
               <p className="text-muted-foreground leading-relaxed whitespace-pre-wrap">
                 {configError || 'Failed to load robot configuration. Please check the backend connection and environment variables.'}
               </p>
               <div className="pt-4">
                 <button
                   onClick={() => window.location.reload()}
                   className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-bold hover:bg-primary/90 transition-colors shadow-lg"
                 >
                   Retry Connection
                 </button>
               </div>
             </div>
           </div>
      )

      }

      <ToastContainer
          position={"top-center"}
          autoClose={5000}
          hideProgressBar={false}
          closeOnClick={false}
          pauseOnHover={true}
          draggable={true}
          theme={"light"}
          transition={Zoom}
      />
    </div>
  )
}
