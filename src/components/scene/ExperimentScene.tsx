"use client";

/**
 * The sampling-space scene (docs/03-design.md):
 * - ground plane = normalized work-zone geography (lon/lat)
 * - height (y) = the oracle's predicted value per record
 * - instanced point cloud: unlabeled dim / labeled solid / new picks pulse
 * - CI volume: translucent slab spanning [ci_lower, ci_upper], tightening
 * - acquisition surface: cumulative selection probability (1 - q_never,
 *   computed by the Python runner — never re-derived here) as a heatmap
 *
 * Rendering contract: instanced meshes only for points; all geometries,
 * materials, and textures created here are disposed on unmount, verified
 * by the e2e memory test via the window.__ppiGlInfo hook.
 */

import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

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

declare global {
  interface Window {
    __ppiGlInfo?: () => { geometries: number; textures: number };
    __ppiFrameTimes?: number[];
  }
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
    // Frame-time samples for the performance gate (bounded buffer).
    const buf = (window.__ppiFrameTimes ??= []);
    if (buf.length < 600) buf.push(delta * 1000);
  });
  return null;
}

function PointCloud({ data }: { data: SceneData }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const pulseRef = useRef(0);
  const { state, meta, freshIdx, reducedMotion, stressPoints = 0 } = data;
  const m = state.data.f_pool.length;
  const total = m + stressPoints;

  const { positions, colors, labeledSet, freshSet } = useMemo(() => {
    const toY = valueScale(state.data.f_pool);
    const pos = new Float32Array(total * 3);
    const x = state.data.x_pool;
    for (let i = 0; i < m; i++) {
      const gx = x ? (x[i]![0]! - 0.5) * WORLD : 0;
      const gz = x ? (x[i]![1]! - 0.5) * WORLD : 0;
      pos[i * 3] = gx;
      pos[i * 3 + 1] = toY(state.data.f_pool[i]!);
      pos[i * 3 + 2] = gz;
    }
    // Deterministic synthetic stress points (frame-rate gate only).
    let s = 1234567;
    const rand = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
    for (let i = m; i < total; i++) {
      pos[i * 3] = (rand() - 0.5) * WORLD;
      pos[i * 3 + 1] = rand() * HEIGHT;
      pos[i * 3 + 2] = (rand() - 0.5) * WORLD;
    }
    return {
      positions: pos,
      colors: new Float32Array(total * 3),
      labeledSet: new Set(state.labeled_idx),
      freshSet: new Set(freshIdx),
    };
  }, [state.data.f_pool, state.data.x_pool, state.labeled_idx, freshIdx, m, total]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const mat = new THREE.Matrix4();
    const color = new THREE.Color();
    const style = getComputedStyle(document.documentElement);
    const cLabeled = new THREE.Color(style.getPropertyValue("--series-ppi").trim() || "#2a78d6");
    const cFresh = new THREE.Color(style.getPropertyValue("--status-warn").trim() || "#eda100");
    const cDim = new THREE.Color(style.getPropertyValue("--ink-3").trim() || "#8a8880");
    for (let i = 0; i < total; i++) {
      const fresh = freshSet.has(i);
      const labeled = labeledSet.has(i);
      const scale = fresh ? 2.2 : labeled ? 1.6 : 1.0;
      mat.makeScale(scale, scale, scale);
      mat.setPosition(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!);
      mesh.setMatrixAt(i, mat);
      color.copy(fresh ? cFresh : labeled ? cLabeled : cDim);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    pulseRef.current = 0;
  }, [positions, labeledSet, freshSet, total]);

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

function CiVolume({ state }: { state: RunState }) {
  const slabRef = useRef<THREE.Mesh>(null);
  const planeRef = useRef<THREE.Mesh>(null);
  const last = state.history[state.history.length - 1];
  const toY = useMemo(() => valueScale(state.data.f_pool), [state.data.f_pool]);
  if (!last?.estimates.ppi) return null;
  const e = last.estimates.ppi;
  const yLo = toY(e.ci_lower);
  const yHi = toY(e.ci_upper);
  const yEst = toY(e.estimate);
  return (
    <group>
      <mesh
        ref={slabRef}
        position={[0, (yLo + yHi) / 2, 0]}
        scale={[1, Math.max(yHi - yLo, 0.02), 1]}
      >
        <boxGeometry args={[WORLD, 1, WORLD]} />
        <meshStandardMaterial color="#2a78d6" transparent opacity={0.14} depthWrite={false} />
      </mesh>
      <mesh ref={planeRef} position={[0, yEst, 0]}>
        <boxGeometry args={[WORLD, 0.015, WORLD]} />
        <meshStandardMaterial color="#2a78d6" transparent opacity={0.55} depthWrite={false} />
      </mesh>
    </group>
  );
}

function AcquisitionSurface({ state }: { state: RunState }) {
  const texture = useMemo(() => {
    // Heatmap of cumulative selection probability 1 - q_never (from the
    // Python runner) binned onto a 64x64 grid over the ground plane.
    const grid = 64;
    const acc = new Float32Array(grid * grid);
    const cnt = new Float32Array(grid * grid);
    const x = state.data.x_pool;
    if (x) {
      state.q_never.forEach((q, i) => {
        const gx = Math.min(grid - 1, Math.max(0, Math.floor(x[i]![0]! * grid)));
        const gy = Math.min(grid - 1, Math.max(0, Math.floor(x[i]![1]! * grid)));
        acc[gy * grid + gx] += 1 - q;
        cnt[gy * grid + gx] += 1;
      });
    }
    const canvas = document.createElement("canvas");
    canvas.width = grid;
    canvas.height = grid;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(grid, grid);
    // Sequential blue ramp (--seq-*); computed here from the same hexes.
    const ramp = [
      [219, 233, 250],
      [156, 195, 238],
      [93, 156, 226],
      [42, 120, 214],
      [23, 84, 143],
    ];
    let maxV = 0;
    for (let i = 0; i < acc.length; i++) {
      if (cnt[i]! > 0) maxV = Math.max(maxV, acc[i]! / cnt[i]!);
    }
    for (let i = 0; i < acc.length; i++) {
      const v = cnt[i]! > 0 && maxV > 0 ? acc[i]! / cnt[i]! / maxV : 0;
      const idx = Math.min(ramp.length - 1, Math.floor(v * ramp.length));
      const [r, g, b] = ramp[idx]!;
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = cnt[i]! > 0 ? 235 : 40;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.magFilter = THREE.NearestFilter;
    return tex;
  }, [state.q_never, state.data.x_pool]);

  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
      <planeGeometry args={[WORLD, WORLD]} />
      <meshBasicMaterial map={texture} side={THREE.DoubleSide} />
    </mesh>
  );
}

export function ExperimentScene({ data }: { data: SceneData }) {
  return (
    <Canvas
      camera={{ position: [8, 6, 8], fov: 45 }}
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
    >
      <GlInfoHook />
      <ambientLight intensity={0.7} />
      <directionalLight position={[6, 10, 4]} intensity={1.1} />
      <PointCloud data={data} />
      <CiVolume state={data.state} />
      <AcquisitionSurface state={data.state} />
      <gridHelper args={[WORLD, 10, "#8a8880", "#55534e"]} position={[0, -0.005, 0]} />
      <OrbitControls
        enableDamping={!data.reducedMotion}
        dampingFactor={0.08}
        maxPolarAngle={Math.PI / 2.05}
        minDistance={3}
        maxDistance={30}
      />
    </Canvas>
  );
}
