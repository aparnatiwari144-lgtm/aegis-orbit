/**
 * AEGIS-ORBIT // Ambient Meteoroid & Cosmic Debris Particle Layer
 *
 * Fully decorative atmospheric visual layer rendered with THREE.InstancedMesh.
 * Purely ambient background particle field with zero data bindings, zero raycast
 * intersections, and zero overlap with active telemetry or tracked-object markers.
 */

(function(root) {
  'use strict';

  const METEOROID_CONFIG = {
    count: 65,                // 40-80 range (optimal balance of density and 60fps performance)
    minRadius: 15.0,          // Outside dense inner LEO shell
    maxRadius: 48.0,          // Loose surrounding atmospheric halo
    baseColor: 0x1d212a,      // Dark matte charcoal stone (distinct from shiny metallic satellites)
    roughness: 0.92,          // Stone-like rough matte finish
    metalness: 0.06,          // Low metalness (reads as asteroid rock, not metal debris)
    heroOpacity: 0.85,        // Full cinematic density during hero intro
    dashboardOpacity: 0.38,   // Ambient subtle density during operational dashboard
    dossierOpacity: 0.22      // Reduced density when telemetry dossier is active
  };

  let instancedMesh = null;
  let meteoroidMaterial = null;
  let rockData = [];
  const dummy = new THREE.Object3D();
  let currentTargetOpacity = METEOROID_CONFIG.heroOpacity;

  /**
   * Generates a procedural high-frequency normal noise map for chiseled rocky roughness.
   */
  function createProceduralRockTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(128, 128);

    for (let i = 0; i < imgData.data.length; i += 4) {
      // Normal vector components encoded into RGB
      const nx = 128 + Math.floor((Math.random() - 0.5) * 45);
      const ny = 128 + Math.floor((Math.random() - 0.5) * 45);
      const nz = 255;
      imgData.data[i] = nx;
      imgData.data[i + 1] = ny;
      imgData.data[i + 2] = nz;
      imgData.data[i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(2, 2);
    return texture;
  }

  /**
   * Initializes and mounts the ambient meteoroid field into the existing Three.js scene.
   * Uses existing directional/ambient scene lighting — introduces zero new light sources.
   *
   * @param {THREE.Scene} scene The existing primary AEGIS-ORBIT scene.
   */
  function initAmbientMeteoroidField(scene) {
    if (!scene || instancedMesh) return;

    // 1. Create deformed low-poly irregular rock geometry
    const baseGeo = new THREE.DodecahedronGeometry(0.7, 1);
    const posAttr = baseGeo.attributes.position;
    const v = new THREE.Vector3();

    for (let i = 0; i < posAttr.count; i++) {
      v.fromBufferAttribute(posAttr, i);
      // Displace vertices organically for chiseled, non-uniform asteroid faceting
      const noise = 1.0 + (Math.sin(v.x * 6.0) * Math.cos(v.y * 6.0) * Math.sin(v.z * 6.0)) * 0.24;
      v.multiplyScalar(noise);
      posAttr.setXYZ(i, v.x, v.y, v.z);
    }
    baseGeo.computeVertexNormals();

    // 2. Rough dark charcoal material with flat-shading for sharp polygon facets
    meteoroidMaterial = new THREE.MeshStandardMaterial({
      color: METEOROID_CONFIG.baseColor,
      roughness: METEOROID_CONFIG.roughness,
      metalness: METEOROID_CONFIG.metalness,
      normalMap: createProceduralRockTexture(),
      normalScale: new THREE.Vector2(0.55, 0.55),
      flatShading: true,
      transparent: true,
      opacity: METEOROID_CONFIG.heroOpacity,
      depthWrite: false // Prevents sorting artifacts with translucent orbit ellipses
    });

    // 3. Built with single InstancedMesh for optimal 60fps GPU batching
    instancedMesh = new THREE.InstancedMesh(baseGeo, meteoroidMaterial, METEOROID_CONFIG.count);
    instancedMesh.name = 'ambientMeteoroidField';
    // Raycaster intentionally ignores this group (purely non-interactive decorative layer)
    instancedMesh.raycast = function() {}; 

    rockData = [];

    for (let i = 0; i < METEOROID_CONFIG.count; i++) {
      // Loose shell distribution around Earth with cinematic depth
      const radius = METEOROID_CONFIG.minRadius + Math.random() * (METEOROID_CONFIG.maxRadius - METEOROID_CONFIG.minRadius);
      const theta = Math.random() * Math.PI * 2;
      const phi = (Math.random() - 0.5) * Math.PI * 0.88;

      const x = radius * Math.cos(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) + (Math.random() - 0.5) * 5.0;
      const z = radius * Math.cos(phi) * Math.sin(theta);

      // Unique non-uniform aspect ratios per rock (no two rocks identical)
      const baseScale = 0.16 + Math.random() * 0.42;
      const sx = baseScale * (0.65 + Math.random() * 0.7);
      const sy = baseScale * (0.65 + Math.random() * 0.7);
      const sz = baseScale * (0.65 + Math.random() * 0.7);

      const rot = new THREE.Euler(
        Math.random() * Math.PI * 2,
        Math.random() * Math.PI * 2,
        Math.random() * Math.PI * 2
      );

      dummy.position.set(x, y, z);
      dummy.rotation.copy(rot);
      dummy.scale.set(sx, sy, sz);
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);

      rockData.push({
        pos: new THREE.Vector3(x, y, z),
        rot: rot,
        scale: new THREE.Vector3(sx, sy, sz),
        rotSpeed: new THREE.Vector3(
          (Math.random() - 0.5) * 0.18,
          (Math.random() - 0.5) * 0.18,
          (Math.random() - 0.5) * 0.18
        ),
        driftVel: new THREE.Vector3(
          (Math.random() - 0.5) * 0.035,
          (Math.random() - 0.5) * 0.020,
          (Math.random() - 0.5) * 0.035
        ),
        initialRadius: radius
      });
    }

    instancedMesh.instanceMatrix.needsUpdate = true;
    scene.add(instancedMesh);
  }

  /**
   * Updates ambient rock tumbles, linear drifts, seamless bounds wrapping, and camera parallax.
   *
   * @param {number} deltaSec Elapsed time delta in seconds.
   * @param {Object} options Scene context flags (isHero, isDossierOpen, camera, mouseParallax).
   */
  function updateAmbientMeteoroidField(deltaSec, options) {
    if (!instancedMesh || !meteoroidMaterial) return;

    const opts = options || {};
    const dt = Math.min(deltaSec || 0.016, 0.1);

    // Dynamic adaptive opacity management
    if (opts.isHero) {
      currentTargetOpacity = METEOROID_CONFIG.heroOpacity;
    } else if (opts.isDossierActive) {
      currentTargetOpacity = METEOROID_CONFIG.dossierOpacity;
    } else {
      currentTargetOpacity = METEOROID_CONFIG.dashboardOpacity;
    }

    // Smooth opacity lerp
    meteoroidMaterial.opacity = THREE.MathUtils.lerp(meteoroidMaterial.opacity, currentTargetOpacity, dt * 3.5);

    // Parallax offset computation based on mouse/camera motion
    const mouseX = opts.mouseParallax ? opts.mouseParallax.x : 0;
    const mouseY = opts.mouseParallax ? opts.mouseParallax.y : 0;

    for (let i = 0; i < rockData.length; i++) {
      const rock = rockData[i];

      // 1. Independent slow tumble rotation
      rock.rot.x += rock.rotSpeed.x * dt;
      rock.rot.y += rock.rotSpeed.y * dt;
      rock.rot.z += rock.rotSpeed.z * dt;

      // 2. Slow linear spatial drift
      rock.pos.x += rock.driftVel.x * dt * 60;
      rock.pos.y += rock.driftVel.y * dt * 60;
      rock.pos.z += rock.driftVel.z * dt * 60;

      // 3. Seamless boundary loop (smooth wrap without abrupt popping)
      const currentDist = rock.pos.length();
      if (currentDist > METEOROID_CONFIG.maxRadius) {
        rock.pos.multiplyScalar(METEOROID_CONFIG.minRadius / currentDist);
      } else if (currentDist < METEOROID_CONFIG.minRadius) {
        rock.pos.multiplyScalar(METEOROID_CONFIG.maxRadius / currentDist);
      }

      // 4. Subtle camera parallax displacement on closest foreground rocks
      let px = rock.pos.x;
      let py = rock.pos.y;
      let pz = rock.pos.z;

      if (currentDist < 26.0 && (mouseX !== 0 || mouseY !== 0)) {
        const parallaxFactor = (26.0 - currentDist) / 26.0 * 0.35;
        px += mouseX * parallaxFactor;
        py += mouseY * parallaxFactor;
      }

      dummy.position.set(px, py, pz);
      dummy.rotation.copy(rock.rot);
      dummy.scale.copy(rock.scale);
      dummy.updateMatrix();

      instancedMesh.setMatrixAt(i, dummy.matrix);
    }

    instancedMesh.instanceMatrix.needsUpdate = true;
  }

  // Expose methods globally for seamless integration
  root.initAmbientMeteoroidField = initAmbientMeteoroidField;
  root.updateAmbientMeteoroidField = updateAmbientMeteoroidField;

})(typeof window !== 'undefined' ? window : this);
