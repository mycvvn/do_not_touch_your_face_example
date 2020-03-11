import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { Howl } from 'howler';
import heyHandSound from './assets/hey_hand.mp3';

var sound = new Howl({
  src: [heyHandSound]
});

const TRAINING_TIMES = 50;

function App() {
  const video = useRef();
  const classifier = useRef();
  const canPlaySound = useRef(true);
  const mobilenetModule = useRef();
  const [touched, setTouched] = useState(false);
  const [message, setMessage] = useState('');

  const setupCamera = () => {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia =
                navigator.getUserMedia ||
                navigator.webkitGetUserMedia ||
                navigator.mozGetUserMedia ||
                navigator.msGetUserMedia;

      if (navigator.getUserMedia) {
        const mediaOptions = {
          video: true
        };
        navigator.getUserMedia(
          mediaOptions,
          stream => {
            video.current.srcObject = stream;
            video.current.addEventListener('loadeddata', resolve);
          },
          error => reject(error)
        );
      } else {
        reject();
      }
    });
  }

  const init = async () => {
    console.log('init');

    setMessage('Đang setup camera...');
    await setupCamera();
    console.log('setup camera success');
    
    classifier.current = knnClassifier.create();

    mobilenetModule.current = await mobilenet.load();
    
    setMessage('Không chạm tay lên mặt và bấm Train');
    console.log('setup done!');
  }

  useEffect(() => {
    init();

    sound.on('end', function() {
      canPlaySound.current = true;
    });

    // cleanup
    return () => {

    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const train = async (label) => {
    setMessage('Đang train cho Machine mặt đẹp trai của bạn...');

    for (let i = 0; i < TRAINING_TIMES; ++i) {
      console.log(`training... (${parseInt((i + 1) / TRAINING_TIMES * 100)}%)`)
      await training(label);
    }
  }

  const training = label => {
    return new Promise(async resolve => {
      const embedding = mobilenetModule.current.infer(
        video.current,
        true
      );
      classifier.current.addExample(embedding, label);

      await sleep(100);
      resolve();
    });
  }

  const sleep = milliseconds => {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
  };

  const run = async () => {
    const embedding = mobilenetModule.current.infer(
      video.current,
      true
    );

    const result = await classifier.current.predictClass(embedding);
    console.log('label: ', result.label)
    console.log('confidence: ', result.confidences[0])

    if (
      result.label === '1' &&
      result.confidences[result.classIndex] > 0.8
    ) {
      setTouched(true);
      if (canPlaySound.current) {
        canPlaySound.current = false;
        sound.play();
      }
    } else {
      setTouched(false);
    }
    
    await sleep(200);
    run();
  }

  console.log('touched', touched)

  return (
    <div className={`wrapper ${touched ? 'touched' : ''}`}>
      <video 
        className="video"
        ref={video}
        autoPlay
      />

      <div className="control">
        <p className="message">{message}</p>
        <button
            className="btn"
            onClick={() =>
              train(0)
            }
          >
            Train 1
          </button>
          <button
            className="btn"
            onClick={() =>
              train(1)
            }
          >
            Train 2
          </button>
          <button
            className="btn"
            onClick={run}
          >
            Ready
          </button>
      </div>
    </div>
  );
}

export default App;
