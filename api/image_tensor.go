package main

import (
    "bytes"

    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func makeTensorFromImage(imageBuffer *bytes.Buffer, imageFormat string) (*tf.Tensor, error) {
    tensor, err := tf.NewTensor(imageBuffer.String())
    if err != nil {
        return nil, err
    }

    graph, input, output, err := makeTensorImageGraph(imageFormat)
    if err != nil {
        return nil, err
    }
    session, err := tf.NewSession(graph, nil)
    if err != nil {
        return nil, err
    }
    defer session.Close()
    normalized, err := session.Run(
        map[tf.Output]*tf.Tensor{input: tensor},
        []tf.Output{output},
        nil)
    if err != nil {
        return nil, err
    }
    return normalized[0], nil
}

func makeTensorImageGraph(imageFormat string) (graph *tf.Graph, input, output tf.Output, err error) {
    const (
        H, W = 244, 244
        Mean = float32(117)
        Scale = float32(1)
    )

    s := op.NewScope()
    input = op.Placeholder(s, tf.String)
    var decode tf.Output
    if imageFormat == "png" {
        decode = op.DecodePng(s, input, op.DecodePngChannels(3))
    } else {
        decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
    }

    output = op.Div(s, 
        op.Sub(s,
            // resize
            op.ResizeBilinear(s,
                // create batch containing single image
                op.ExpandDims(s,
                    // use decoded pixel values
                    op.Cast(s, decode, tf.Float),
                    op.Const(s.SubScope("make_batch"), int32(0))),
                op.Const(s.SubScope("size"), []int32{H, W})),
            op.Const(s.SubScope("mean"), Mean)),
        op.Const(s.SubScope("scale"), Scale))
    graph, err = s.Finalize()
    return graph, input, output, err
}