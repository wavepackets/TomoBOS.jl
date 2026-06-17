module TomoBOS

using PythonCall

function test_opencv()
    cv2 = pyimport("cv2")
    println("OpenCV version: ", cv2.__version__)
end

export test_opencv

end
