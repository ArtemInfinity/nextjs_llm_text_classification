import { HfInference, TextClassificationOutput } from "@huggingface/inference";

let hf: HfInference;

export async function POST(req: Request, res: Response){
  
  const {input} = await req.json()
  const inferenceResponse: TextClassificationOutput = await runInference(input)
  /* 
  [
    { label: 'joy', score: 0.8315524458885193 },
    { label: 'admiration', score: 0.08930609375238419 },
    { label: 'approval', score: 0.06290879845619202 },
    { label: 'relief', score: 0.05792760103940964 },
    { label: 'neutral', score: 0.045840729027986526 },
    { label: 'caring', score: 0.02632378600537777 },
  ]
  */
  const filteredResponse = filterResponses([...inferenceResponse])

  return new Response(JSON.stringify({
    inferenceResponse,
    filteredResponse
  }), {status: 200})
}

async function runInference(input: string){
  if(!hf){
    hf = new HfInference(process.env.HF_TOKEN)
  }

  const modelName = 'SamLowe/roberta-base-go_emotions'
  const inferenceRes = await hf.textClassification({
    model: modelName,
    inputs: input
  })
  return inferenceRes;
}

function filterResponses(emotions: TextClassificationOutput){
  const filtered = []
  const emotion0 = emotions.shift()
  filtered.push(emotion0)
  let score = emotion0?.score

  while(emotions.length > 0){
    const emotionI = emotions.shift()
    if(emotionI?.score! > score!*0.5){
      filtered.push(emotionI)
      score = emotionI?.score
    } else {
      break;
    }
  }
  return filtered
}