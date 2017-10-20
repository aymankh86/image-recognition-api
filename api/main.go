package main

import (
    "bufio"
    "bytes"
    "io"
    "io/ioutil"
    "log"
    "net/http"
    "os"
    "sort"
    "strings"

    "github.com/julienschmidt/httprouter"
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type ClassifyResult struct {
    Filename string         `json:"filename"`
    Labels   []LabelResult  `json:"labels`
}

type LabelResult struct {
    Label       string      `json:"label"`
    Probability float32     `json:"probability`
}

var (
    graph *tf.Graph
    labels []string
)

func main() {
    if err := loadModel(); err != nil {
        log.Fatal(err)
        return
    }

    r := httprouter.New()
    r.POST("/recognize", recognizeHandler)
    log.Fatal(http.ListenAndServe(":8080", r))
}

func loadModel() error {
    model, err := ioutil.ReadFile("/model/tensorflow_inception_graph.pb")
    if err != nil {
        return err
    }
    graph = tf.NewGraph()
    if err := graph.Import(model, ""); err != nil {
        return err
    }

    labelsFile, err := os.Open("/model/imagenet_comp_graph_label_strings.txt")
    if err != nil {
        return err
    }
    defer labelsFile.Close()
    scanner := bufio.NewScanner(labelsFile)
    for scanner.Scan() {
        labels = append(labels, scanner.Text())
    }
    if err := scanner.Err(); err != nil {
        return err
    }
    return nil
}

func recognizeHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
    imageFile, header, err := r.FormFile("image")
    imageName := strings.Split(header.Filename, ".")
    if err != nil {
        responseError(w, "Could not read image", http.StatusBadRequest)
        return
    }
    defer imageFile.Close()
    var imageBuffer bytes.Buffer
    io.Copy(&imageBuffer, imageFile)

    tensor, err := makeTensorFromImage(&imageBuffer, imageName[:1][0])
    if err != nil {
        responseError(w, "Invalid image", http.StatusBadRequest)
        return
    }

    session, err := tf.NewSession(graph, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer session.Close()
    output, err := session.Run(
        map[tf.Output]*tf.Tensor{
            graph.Operation("input").Output(0): tensor,
        },
        []tf.Output{
            graph.Operation("output").Output(0),
        },
        nil)
    if err != nil {
        responseError(w, "Could not run interface", http.StatusInternalServerError)
        return
    }
    responseJSON(w, ClassifyResult{
        Filename: header.Filename,
        Labels: findBestLabels(output[0].Value().([][]float32)[0]),
    })
}

type ByProbability []LabelResult
func (a ByProbability) Len() int            {return len(a)}
func (a ByProbability) Swap(i, j int)       {a[i], a[j] = a[j], a[i]}
func (a ByProbability) Less(i, j int) bool  {return a[i].Probability > a[j].Probability}

func findBestLabels(probabilities []float32) []LabelResult {
    var resultLabels []LabelResult
    for i, p := range probabilities {
        if i >= len(labels) {
            break
        }
        resultLabels = append(resultLabels, LabelResult{Label: labels[i], Probability: p})
    }
    sort.Sort(ByProbability(resultLabels))
    return resultLabels[:5]
}