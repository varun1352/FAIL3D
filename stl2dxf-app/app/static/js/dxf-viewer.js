/**
 * DXF Viewer class for the STL to DXF Converter application
 */
class DXFViewer {
    constructor(container) {
        if (!container) {
            throw new Error('Container element is required');
        }
        
        this.container = container;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111); // Darker background
        this.camera = new THREE.PerspectiveCamera(
            75,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 10);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = true;
        this.grid = new THREE.GridHelper(100, 100, 0x333333, 0x222222);
        this.grid.material.opacity = 0.2;
        this.grid.material.transparent = true;
        this.scene.add(this.grid);
        this.measureMode = false;
        this.measurePoints = [];
        this.measureLine = null;
        this.measureText = null;
        this.layers = new Map();
        this.layerColors = {
            default: 0xffffff,
            dimensions: 0x3b82f6, // Bright blue to match logo
            hidden: 0x666666
        };
        window.addEventListener('resize', this.onWindowResize.bind(this));
        this.renderer.domElement.addEventListener('click', this.onCanvasClick.bind(this));
        this.animate();
    }
    
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    loadDXF(dxfData) {
        const parser = new DXFParser();
        const dxf = parser.parseSync(dxfData);
        this.clearScene();
        if (dxf.entities && Array.isArray(dxf.entities)) {
            dxf.entities.forEach(entity => {
                try {
                    this.processEntity(entity);
                } catch (err) {
                    console.warn('Error processing entity:', err);
                }
            });
        }
        this.updateLayerList();
        this.centerView();
    }
    
    processEntity(entity) {
        let geometry;
        let material;
        let mesh;
        const layerColor = this.getLayerColor(entity.layer);
        material = new THREE.LineBasicMaterial({ color: layerColor });
        switch (entity.type) {
            case 'LINE':
                geometry = new THREE.BufferGeometry();
                const points = [
                    new THREE.Vector3(entity.start.x, entity.start.y, 0),
                    new THREE.Vector3(entity.end.x, entity.end.y, 0)
                ];
                geometry.setFromPoints(points);
                mesh = new THREE.Line(geometry, material);
                break;
            case 'CIRCLE':
                geometry = new THREE.CircleGeometry(entity.radius, 32);
                mesh = new THREE.Line(geometry, material);
                mesh.position.set(entity.center.x, entity.center.y, 0);
                break;
            case 'ARC':
                geometry = new THREE.CircleGeometry(
                    entity.radius,
                    32,
                    entity.startAngle,
                    entity.angleLength
                );
                mesh = new THREE.Line(geometry, material);
                mesh.position.set(entity.center.x, entity.center.y, 0);
                break;
            case 'POLYLINE':
                if (entity.vertices && entity.vertices.length > 1) {
                    geometry = new THREE.BufferGeometry();
                    const points = entity.vertices.map(v => 
                        new THREE.Vector3(v.x, v.y, 0)
                    );
                    geometry.setFromPoints(points);
                    mesh = new THREE.Line(geometry, material);
                }
                break;
            default:
                console.log('Unsupported entity type:', entity.type);
                return;
        }
        if (mesh) {
            let layer = this.layers.get(entity.layer);
            if (!layer) {
                layer = new THREE.Group();
                layer.name = entity.layer;
                this.layers.set(entity.layer, layer);
                this.scene.add(layer);
            }
            layer.add(mesh);
        }
    }
    
    updateLayerList() {
        const layerList = document.getElementById('layer-list');
        layerList.innerHTML = '';
        this.layers.forEach((group, layerName) => {
            const layerDiv = document.createElement('div');
            layerDiv.className = 'flex items-center justify-between p-2 bg-secondary rounded-md';
            const label = document.createElement('span');
            label.textContent = layerName;
            label.className = 'text-secondary-foreground';
            const toggle = document.createElement('input');
            toggle.type = 'checkbox';
            toggle.checked = group.visible;
            toggle.className = 'toggle';
            toggle.addEventListener('change', (e) => {
                group.visible = e.target.checked;
            });
            layerDiv.appendChild(label);
            layerDiv.appendChild(toggle);
            layerList.appendChild(layerDiv);
        });
    }
    
    toggleMeasureMode() {
        this.measureMode = !this.measureMode;
        if (!this.measureMode) {
            this.clearMeasurement();
        }
    }
    
    onCanvasClick(event) {
        if (!this.measureMode) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);
        const intersects = raycaster.intersectObjects(this.scene.children, true);
        if (intersects.length > 0) {
            const point = intersects[0].point;
            this.addMeasurePoint(point);
        }
    }
    
    addMeasurePoint(point) {
        this.measurePoints.push(point);
        if (this.measurePoints.length === 2) {
            this.showMeasurement();
            this.measurePoints = [];
        }
    }
    
    showMeasurement() {
        const start = this.measurePoints[0];
        const end = this.measurePoints[1];
        const distance = start.distanceTo(end);
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const material = new THREE.LineBasicMaterial({ 
            color: this.layerColors.dimensions 
        });
        if (this.measureLine) {
            this.scene.remove(this.measureLine);
        }
        this.measureLine = new THREE.Line(geometry, material);
        this.scene.add(this.measureLine);
        const measurementsList = document.getElementById('measurements');
        const measurement = document.createElement('div');
        measurement.className = 'flex justify-between items-center p-2';
        measurement.innerHTML = `
            <span>Distance:</span>
            <span>${distance.toFixed(2)} units</span>
        `;
        measurementsList.appendChild(measurement);
    }
    
    clearMeasurement() {
        if (this.measureLine) {
            this.scene.remove(this.measureLine);
            this.measureLine = null;
        }
        this.measurePoints = [];
    }
    
    getLayerColor(layerName) {
        return this.layerColors[layerName] || this.layerColors.default;
    }
    
    clearScene() {
        while(this.scene.children.length > 0) { 
            this.scene.remove(this.scene.children[0]); 
        }
        this.layers.clear();
        this.scene.add(this.grid);
    }
    
    centerView() {
        const box = new THREE.Box3();
        this.scene.traverse((obj) => {
            if (obj.geometry) {
                obj.geometry.computeBoundingBox();
                box.expandByObject(obj);
            }
        });
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y);
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.5; // Add some padding
        this.camera.position.set(center.x, center.y, cameraZ);
        this.controls.target.set(center.x, center.y, 0);
        this.camera.updateProjectionMatrix();
        this.controls.update();
    }
    
    zoomIn() {
        this.camera.position.z *= 0.9;
    }
    
    zoomOut() {
        this.camera.position.z *= 1.1;
    }
    
    resetView() {
        this.centerView();
    }
}