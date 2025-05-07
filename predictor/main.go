package main

import (
	"embed"
	_ "embed"
	"fmt"
	"log"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

//go:embed model.onnx
var modelFS embed.FS

func main() {
	// モデルファイルの読み込み
	modelData, err := modelFS.ReadFile("model.onnx")
	if err != nil {
		log.Fatalf("モデルファイルの読み込みに失敗: %v", err)
	}

	// ONNX モデルの作成
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)

	// モデルのデシリアライズ
	err = model.UnmarshalBinary(modelData)
	if err != nil {
		log.Fatalf("モデルのデシリアライズに失敗: %v", err)
	}

	input := float32(20.0)

	// 入力テンソルの作成 (x = 3.0 のケース)
	inputTensor := tensor.New(tensor.WithShape(1, 1), tensor.WithBacking([]float32{input / 10.0}))

	// 推論の実行
	err = model.SetInput(0, inputTensor)
	if err != nil {
		log.Fatalf("入力の設定に失敗: %v", err)
	}

	err = backend.Run()
	if err != nil {
		log.Fatalf("推論の実行に失敗: %v", err)
	}

	// 出力の取得
	output, err := model.GetOutputTensors()
	if err != nil {
		log.Fatalf("出力の取得に失敗: %v", err)
	}

	// 結果の表示
	fmt.Printf("Input x=%f, Predicted: %v\n", input, output[0])
}
