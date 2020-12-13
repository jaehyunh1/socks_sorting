const express = require('express');
const router = express.Router();
const { PythonShell } = require("python-shell");

router.get('/', (req, res) => {
    console.log('http://localhost:3002/api/');
    
    let options = {
        pythonPath : 'python',
    };

    PythonShell.run('test.py', options, (err, result) => {
        if (err) throw err;
        console.log(`result: ${result}`);
        res.send({title: result.toString()});
    });
   
});

module.exports = router;