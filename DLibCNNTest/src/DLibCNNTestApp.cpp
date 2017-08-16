// http://dlib.net/dnn_introduction_ex.cpp.html

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ip/Resize.h"

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

using namespace ci;
using namespace ci::app;
using namespace std;

using namespace dlib;

using dlibImageArrayGrayscale = array2d<unsigned char>;

using net_type = loss_multiclass_log <
                 fc<10,
                 relu<fc<84,
                 relu<fc<120,
                 max_pool<2, 2, 2, 2, relu<con<16, 5, 5, 1, 1,
                 max_pool<2, 2, 2, 2, relu<con<6, 5, 5, 1, 1,
                 input<matrix<unsigned char>>
                 >>>>>>>>>> >>;

class DLibCNNTestApp : public App
{
	public:
		void setup() override;

		void mouseDrag( MouseEvent event ) override;

		void keyDown( KeyEvent event ) override;
		void update() override;
		void draw() override;

		dlibImageArrayGrayscale getDlibImageArrayForCiChannel( ci::Channel32fRef channel );

		net_type mNet;
		int mLastDigit;
		ci::Font mFont;
		std::vector<glm::vec2> mPoints;
};

void DLibCNNTestApp::setup()
{
	// train and save to disk
	std::vector<matrix<unsigned char>> training_images;
	std::vector<unsigned long>         training_labels;
	std::vector<matrix<unsigned char>> testing_images;
	std::vector<unsigned long>         testing_labels;
	load_mnist_dataset( getAssetPath( "data" ).string(), training_images, training_labels, testing_images, testing_labels );

	ci::app::console() << testing_images.at( 0 ).NC << std::endl;
	ci::app::console() << testing_images.at( 0 ).NR << std::endl;

	//dnn_trainer<net_type> trainer( mNet );
	//trainer.set_learning_rate( 0.01 );
	//trainer.set_min_learning_rate( 0.00001 );
	//trainer.set_mini_batch_size( 128 );
	//trainer.be_verbose();

	//trainer.set_synchronization_file( "../assets/mnist_sync", std::chrono::seconds( 20 ) );
	//trainer.train( training_images, training_labels );
	//net.clean();
	//serialize( "../assets/mnist_network.dat" ) << net;

	// recall from disk
	deserialize( "../assets/mnist_network.dat" ) >> mNet;

	ci::gl::color( ci::ColorAf( 1.0f, 1.0f, 1.0f, 1.0f ) );
	mLastDigit = -1;
	mFont = ci::Font( "Verdana", 20.0f );

	gl::disableDepthRead();
}

void DLibCNNTestApp::mouseDrag( MouseEvent event )
{
	mPoints.push_back( event.getPos() );
}

void DLibCNNTestApp::keyDown( KeyEvent event )
{
	if( event.getChar() == KeyEvent::KEY_s ) {
		// capture window surface and convert to 28 x 28 to match training image size (MNIST)
		auto capture = copyWindowSurface();
		ci::Surface8u resizedSurface = ci::ip::resizeCopy( capture, capture.getBounds(), glm::vec2( 28, 28 ) );
		ci::Channel32fRef captureChannel = ci::Channel32f::create( resizedSurface );
		//writeImage( "../assets/digit.png", resizedSurface );

		auto dLibArray = getDlibImageArrayForCiChannel( captureChannel );
		dlib::matrix<unsigned char> dLibMat = dlib::mat( dLibArray );

		mLastDigit = mNet( dLibMat );
		mPoints.clear();
	}
	else if( event.getChar() == KeyEvent::KEY_c ) {
	}
}

// Convert Cinder Channel to dlib grayscale image array
dlibImageArrayGrayscale DLibCNNTestApp::getDlibImageArrayForCiChannel( ci::Channel32fRef channel )
{
	// Create dlib array of unsigned chars (grayscale)
	dlib::array2d<unsigned char> dlibImageArray( channel->getSize().y, channel->getSize().x );

	// Iterate through channel and copy in data
	ci::Channel32f::Iter iter = channel->getIter();

	int row = 0;
	int column = 0;

	while( iter.line() ) {
		while( iter.pixel() ) {
			dlibImageArray[row][column] = iter.v() * 255.0f;

			column++;
		}

		row++;
		column = 0;
	}

	// Return the dlib image array
	return dlibImageArray;
}

void DLibCNNTestApp::update()
{
}

void DLibCNNTestApp::draw()
{
	gl::clear( ColorAf( 0.0f, 0.0f, 0.0f, 1.0f ) );

	for( auto point : mPoints ) {
		ci::gl::drawSolidCircle( point, 8.0f );
	}

	gl::drawString( std::to_string( mLastDigit ), glm::vec2( 20.0f, 20.0f ), ci::Color::white(), mFont );
}

CINDER_APP( DLibCNNTestApp, RendererGl )
