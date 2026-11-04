import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function httpToWs(url: string) {
  if (!url) return ''
  return url.replace(/^http/, 'ws')
}

export function formatTimestamp(ns?: number) {
  if (!ns || ns === 0) return ''
  const ms = ns / 1_000_000
  const date = new Date(ms)

  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  const hh = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  const ss = date.getSeconds()
  const fraction = Math.floor((ms % 1000) / 10).toString().padStart(2, '0')

  return `${y}-${m}-${d} ${hh}:${mm}:${String(ss).padStart(2, '0')}.${fraction}`
}
