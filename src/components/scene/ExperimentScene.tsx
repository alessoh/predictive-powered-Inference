"use client";

/**
 * The sampling-space scene (docs/03-design.md):
 * - ground plane = normalized work-zone geography (lon/lat)
 * - height (y) = the oracle's predicted value per record
 * - instanced point cloud: unlabeled dim / labeled solid / new picks pulse
 * - CI volume: translucent slab spanning [ci_lower, ci_upper], animated
 *   toward its new extent with a ~200ms ease (off under reduced motion)
 * - acquisition surface: cumulative selection probability (1 - q_never,
 *   computed by the Python runner — never re-derived here) as a heatmap
 *   with an on-screen legend giving the actual probability scale
 *
 * All colors come from the design tokens (CSS custom properties) so the
 * scene follows light/dark like the rest of the app. Double-click resets
 * the camera. Geometries/materials/textures dispose on unmount, verified
 * by the e2e memory test via the window.__ppiGlInfo hook.
 */

import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { fmtInterval } from "@/components/StatTiles";
import type { PoolRecordMeta, RunState } from "@/lib/run-state";

export interface SceneData {
  state: RunState;
  meta: PoolRecordMeta[];
  /** Indices labeled in the most recent round (for the pulse). */
  freshIdx: number[];
  reducedMotion: boolean;
  /** Synthetic extra points for the frame-time gate (?stress=N). */
  stressPoints?: number;
}

const WORLD = 10; // ground plane spans [-5, 5] in x/z
const HEIGHT = 4; // value axis maps into [0, HEIGHT]
const HOME_POSITION: [number, number, number] = [8, 6, 8];

declare global {
  interface Window {
    __ppiGlInfo?: () => { geometries: number; textures: number };
    __ppiFrameTimes?: number[];
  }
}

/** Design tokens resolved from CSS custom properties at mount, so the
 * scene follows the active theme (review C3). */
interface SceneTheme {
  labeled: string;
  fresh: string;
  dim: string;
  accent: string;
  ramp: [number, number, number][];
  gridMajor: string;
  gridMinor: string;
  ink: string;
}

function readTheme(): SceneTheme {
  const style = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  const hexToRgb = (hex: string): [number, number, number] => {
    const m = hex.replace("#", "");
    return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
  };
  return {
    labeled: v("--series-ppi", "#2a78d6"),
    fresh: v("--status-warn", "#eda100"),
    dim: v("--ink-3", "#54524d"),
    accent: v("--series-ppi", "#2a78d6"),
    ramp: [
      hexToRgb(v("--seq-1", "#dbe9fa")),
      hexToRgb(v("--seq-2", "#9cc3ee")),
      hexToRgb(v("--seq-3", "#5d9ce2")),
      hexToRgb(v("--seq-4", "#2a78d6")),
      hexToRgb(v("--seq-5", "#17548f")),
    ],
    gridMajor: v("--ink-3", "#54524d"),
    gridMinor: v("--edge", "#dedcd6"),
    ink: v("--ink-1", "#0b0b0b"),
  };
}

function valueScale(f: number[]): (v: number) => number {
  const lo = Math.min(...f);
  const hi = Math.max(...f);
  const span = hi - lo || 1;
  return (v: number) => ((v - lo) / span) * HEIGHT;
}

function GlInfoHook() {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    window.__ppiGlInfo = () => ({
      geometries: gl.info.memory.geometries,
      textures: gl.info.memory.textures,
    });
    return () => {
      delete window.__ppiGlInfo;
    };
  }, [gl]);
  useFrame((_, delta) => {
    const buf = (window.__ppiFrameTimes ??= []);
    if (buf.length < 600) buf.push(delta * 1000);
  });
  return null;
}

/** Double-click resets the camera to home (review C5). */
function CameraReset({ controls }: { controls: React.RefObject<OrbitControlsImpl | null> }) {
  const { camera, gl } = useThree();
  useEffect(() => {
    const onDblClick = () => {
      camera.position.set(...HOME_POSITION);
      controls.current?.target.set(0, HEIGHT / 2, 0);
      controls.current?.update();
    };
    gl.domElement.addEventListener("dblclick", onDblClick);
    return () => gl.domElement.removeEventListener("dblclick", onDblClick);
  }, [camera, gl, controls]);
  return null;
}

function PointCloud({ data, theme }: { data: SceneData; theme: SceneTheme }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const pulseRef = useRef(0);
  const { state, freshIdx, reducedMotion, stressPoints = 0 } = data;
  const m = state.data.f_pool.length;
  const total = m + stressPoints;

  const { positions, labeledSet, freshSet } = useMemo(() => {
    const toY = valueScale(state.data.f_pool);
    const pos = new Float32Array(total * 3);
    const x = state.data.x_pool;
    for (let i = 0; i < m; i++) {
      pos[i * 3] = x ? (x[i]![0]! - 0.5) * WORLD : 0;
      pos[i * 3 + 1] = toY(state.data.f_pool[i]!);
      pos[i * 3 + 2] = x ? (x[i]![1]! - 0.5) * WORLD : 0;
    }
    // Deterministic synthetic stress points (frame-rate gate only).
    let s = 1234567;
    const rand = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let i = m; i < total; i++) {
      pos[i * 3] = (rand() - 0.5) * WORLD;
      pos[i * 3 + 1] = rand() * HEIGHT;
      pos[i * 3 + 2] = (rand() - 0.5) * WORLD;
    }
    return {
      positions: pos,
      labeledSet: new Set(state.labeled_idx),
      freshSet: new Set(freshIdx),
    };
  }, [state.data.f_pool, state.data.x_pool, state.labeled_idx, freshIdx, m, total]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const mat = new THREE.Matrix4();
    const color = new THREE.Color();
    const cLabeled = new THREE.Color(theme.labeled);
    const cFresh = new THREE.Color(theme.fresh);
    const cDim = new THREE.Color(theme.dim);
    for (let i = 0; i < total; i++) {
      const fresh = freshSet.has(i);
      const labeled = labeledSet.has(i);
      const scale = fresh ? 2.2 : labeled ? 1.6 : 1.0;
      mat.makeScale(scale, scale, scale);
      mat.setPosition(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!);
      mesh.setMatrixAt(i, mat);
      color.set(fresh ? cFresh : labeled ? cLabeled : cDim);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    pulseRef.current = 0;
  }, [positions, labeledSet, freshSet, total, theme]);

  useFrame((_, delta) => {
    // One-shot pulse on freshly selected points (never loops).
    if (reducedMotion || freshSet.size === 0 || pulseRef.current > 0.6) return;
    pulseRef.current += delta;
    const mesh = meshRef.current;
    if (!mesh) return;
    const k = 1 + 0.6 * Math.sin((pulseRef.current / 0.6) * Math.PI);
    const mat = new THREE.Matrix4();
    for (const i of freshSet) {
      mat.makeScale(2.2 * k, 2.2 * k, 2.2 * k);
      mat.setPosition(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!);
      mesh.setMatrixAt(i, mat);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, total]} frustumCulled={false}>
      <sphereGeometry args={[0.035, 8, 8]} />
      <meshStandardMaterial roughness={0.6} />
    </instancedMesh>
  );
}

function CiVolume({
  state,
  theme,
  reducedMotion,
}: {
  state: RunState;
  theme: SceneTheme;
  reducedMotion: boolean;
}) {
  const slabRef = useRef<THREE.Mesh>(null);
  const planeRef = useRef<THREE.Mesh>(null);
  const last = state.history[state.history.length - 1];
  const toY = useMemo(() => valueScale(state.data.f_pool), [state.data.f_pool]);
  const e = last?.estimates.ppi;
  const target = useMemo(
    () =>
      e
        ? {
            mid: (toY(e.ci_lower) + toY(e.ci_upper)) / 2,
            h: Math.max(toY(e.ci_upper) - toY(e.ci_lower), 0.02),
            est: toY(e.estimate),
          }
        : null,
    [e, toY],
  );

  // ~200ms ease-out toward the new interval (design motion spec); jumps
  // straight to the target under reduced motion.
  useFrame((_, delta) => {
    if (!target || !slabRef.current || !planeRef.current) return;
    const k = reducedMotion ? 1 : Math.min(1, delta / 0.2);
    const slab = slabRef.current;
    slab.position.y += (target.mid - slab.position.y) * k;
    slab.scale.y += (target.h - slab.scale.y) * k;
    planeRef.current.position.y += (target.est - planeRef.current.position.y) * k;
  });

  if (!e || !target) return null;
  return (
    <group>
      <mesh ref={slabRef} position={[0, target.mid, 0]} scale={[1, target.h, 1]}>
        <boxGeometry args={[WORLD, 1, WORLD]} />
        <meshStandardMaterial color={theme.accent} transparent opacity={0.14} depthWrite={false} />
      </mesh>
      <mesh ref={planeRef} position={[0, target.est, 0]}>
        <boxGeometry args={[WORLD, 0.015, WORLD]} />
        <meshStandardMaterial color={theme.accent} transparent opacity={0.55} depthWrite={false} />
      </mesh>
    </group>
  );
}

/** Heatmap statistics shared with the HTML legend. */
export function acquisitionStats(state: RunState): { maxCellMean: number } {
  const grid = 64;
  const acc = new Float32Array(grid * grid);
  const cnt = new Float32Array(grid * grid);
  const x = state.data.x_pool;
  if (x) {
    state.q_never.forEach((q, i) => {
      const gx = Math.min(grid - 1, Math.max(0, Math.floor(x[i]![0]! * grid)));
      const gy = Math.min(grid - 1, Math.max(0, Math.floor(x[i]![1]! * grid)));
      const cell = gy * grid + gx;
      acc[cell] = (acc[cell] ?? 0) + (1 - q);
      cnt[cell] = (cnt[cell] ?? 0) + 1;
    });
  }
  let maxCellMean = 0;
  for (let i = 0; i < acc.length; i++) {
    const c = cnt[i] ?? 0;
    if (c > 0) maxCellMean = Math.max(maxCellMean, (acc[i] ?? 0) / c);
  }
  return { maxCellMean };
}

function AcquisitionSurface({ state, theme }: { state: RunState; theme: SceneTheme }) {
  const texture = useMemo(() => {
    const grid = 64;
    const acc = new Float32Array(grid * grid);
    const cnt = new Float32Array(grid * grid);
    const x = state.data.x_pool;
    if (x) {
      state.q_never.forEach((q, i) => {
        const gx = Math.min(grid - 1, Math.max(0, Math.floor(x[i]![0]! * grid)));
        const gy = Math.min(grid - 1, Math.max(0, Math.floor(x[i]![1]! * grid)));
        const cell = gy * grid + gx;
        acc[cell] = (acc[cell] ?? 0) + (1 - q);
        cnt[cell] = (cnt[cell] ?? 0) + 1;
      });
    }
    const { maxCellMean } = acquisitionStats(state);
    const canvas = document.createElement("canvas");
    canvas.width = grid;
    canvas.height = grid;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(grid, grid);
    const ramp = theme.ramp;
    for (let i = 0; i < acc.length; i++) {
      const c = cnt[i] ?? 0;
      const v = c > 0 && maxCellMean > 0 ? (acc[i] ?? 0) / c / maxCellMean : 0;
      const idx = Math.min(ramp.length - 1, Math.floor(v * ramp.length));
      const [r, g, b] = ramp[idx] ?? ramp[0]!;
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = c > 0 ? 235 : 40;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.magFilter = THREE.NearestFilter;
    return tex;
  }, [state, theme]);

  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
      <planeGeometry args={[WORLD, WORLD]} />
      <meshBasicMaterial map={texture} side={THREE.DoubleSide} />
    </mesh>
  );
}

export function ExperimentScene({ data }: { data: SceneData }) {
  // Lazy initial read (client-only component, rendered via next/dynamic
  // ssr:false); the media-query listener re-reads on OS theme changes.
  const [theme, setTheme] = useState<SceneTheme | null>(() =>
    typeof window === "undefined" ? null : readTheme(),
  );
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setTheme(readTheme());
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  const stats = useMemo(() => acquisitionStats(data.state), [data.state]);
  const last = data.state.history[data.state.history.length - 1];
  const e = last?.estimates.ppi;
  // One precision across hi/est/lo (same rule as the stat tiles).
  const fmtV = e ? fmtInterval([e.estimate, e.ci_lower, e.ci_upper]) : (v: number) => String(v);
  if (!theme) return null;

  return (
    <div className="flex h-full flex-col">
      <div className="relative min-h-0 grow">
        {e && (
          <div
            className="num pointer-events-none absolute top-2 right-2 z-10 rounded-md px-2 py-1 text-right text-[11px]"
            style={{ background: "var(--surface-2)", border: "1px solid var(--edge)" }}
            aria-label="Confidence interval slab values"
          >
            <div style={{ color: "var(--ink-2)" }}>slab: 95% CI</div>
            <div>hi {fmtV(e.ci_upper)}</div>
            <div style={{ color: "var(--series-ppi)" }}>est {fmtV(e.estimate)}</div>
            <div>lo {fmtV(e.ci_lower)}</div>
          </div>
        )}
        <Canvas
          camera={{ position: HOME_POSITION, fov: 45 }}
          dpr={[1, 2]}
          gl={{ antialias: true, powerPreference: "high-performance" }}
        >
          <GlInfoHook />
          <CameraReset controls={controlsRef} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[6, 10, 4]} intensity={1.1} />
          <PointCloud data={data} theme={theme} />
          <CiVolume state={data.state} theme={theme} reducedMotion={data.reducedMotion} />
          <AcquisitionSurface state={data.state} theme={theme} />
          <gridHelper
            args={[WORLD, 10, theme.gridMajor, theme.gridMinor]}
            position={[0, -0.005, 0]}
          />
          <OrbitControls
            ref={controlsRef}
            enableDamping={!data.reducedMotion}
            dampingFactor={0.08}
            maxPolarAngle={Math.PI / 2.05}
            minDistance={3}
            maxDistance={30}
            target={[0, HEIGHT / 2, 0]}
          />
        </Canvas>
      </div>
      {/* Acquisition legend: the actual probability scale, with the
          normalization disclosed (review C2). */}
      <div
        className="flex items-center gap-2 px-2 py-1 text-[11px]"
        style={{ color: "var(--ink-2)" }}
        aria-label="Acquisition heatmap legend"
      >
        <span className="num">0</span>
        <span
          aria-hidden
          style={{
            display: "inline-block",
            width: 96,
            height: 8,
            borderRadius: 4,
            background: `linear-gradient(to right, var(--seq-1), var(--seq-2), var(--seq-3), var(--seq-4), var(--seq-5))`,
          }}
        />
        <span className="num">{stats.maxCellMean.toFixed(3)}</span>
        <span>
          mean cumulative selection probability per cell (1 − q<sub>never</sub>), scaled to the
          maximum cell · double-click resets the camera
        </span>
      </div>
    </div>
  );
}
