

function render(id, objpath, shade) {
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x8FBCD4);
    

    var renderer = new THREE.WebGLRenderer();
    var width = document.querySelector("#anky_image").scrollWidth * 0.9;
    width = Math.max(width, window.innerWidth * 0.5);
    var height = document.querySelector("#anky_image").scrollHeight * 0.9;
    height = Math.max(height, window.innerHeight * 0.5);
    var camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    renderer.setSize(width, height);

    renderer.gammaInput = true;
    renderer.gammaOutput = true;

    document.querySelector(id).appendChild(renderer.domElement);

    var controls = new THREE.OrbitControls(camera, renderer.domElement);

    var material = new THREE.MeshPhongMaterial(
    {color: 0xffffff,
     shading: shade});

    var lightHolder = new THREE.Group();

    var keyLight = new THREE.DirectionalLight(new THREE.Color('hsl(30, 100%, 75%)'), 1.0);
    keyLight.position.set(-100, 0, 100);

    var fillLight = new THREE.DirectionalLight(new THREE.Color('hsl(240, 100%, 75%)'), 0.75);    
    fillLight.position.set(100, 0, 100);

    var backLight = new THREE.DirectionalLight(0xffffff, 1.0);
    backLight.position.set(100, 0, -100).normalize();

    lightHolder.add(keyLight);
    lightHolder.add(fillLight);
    lightHolder.add(backLight);
    scene.add(lightHolder);
    
   
    function callbackOnLoad(object3d) {
        object3d.receiveShadow = true;
        object3d.castShadow = true;
        object3d.traverse( function ( child ) {

        if ( child instanceof THREE.Mesh ) {
            child.material = material;
        }

    } );

        scene.add(object3d);
    }

    var loader = new THREE.OBJLoader();
    loader.load(objpath, callbackOnLoad, null, null, null);

    camera.position.z = 1;

    console.log(scene.children);

    var animate = function () {
        requestAnimationFrame(animate);
        controls.update();
        lightHolder.quaternion.copy(camera.quaternion);
        renderer.render(scene, camera);
    };

    animate();
}
