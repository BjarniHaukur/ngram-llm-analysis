use hyper::{Body, Request, Response, Server};
use hyper::service::{make_service_fn, service_fn};
use std::convert::Infallible;

async fn hello_world(_req: Request<Body>) -> Result<Response<Body>, Infallible> {
    Ok(Response::new(Body::from("Hello, world!")))
}

#[tokio::main]
async fn main() {
    // Define the address to bind the server to
    let addr = ([0, 0, 0, 0], 3000).into(); 


    // Create a service that responds to every request with "Hello, world!"
    let make_svc = make_service_fn(|_conn| async {
        Ok::<_, Infallible>(service_fn(hello_world))
    });

    // Create the server
    let server = Server::bind(&addr).serve(make_svc);

    println!("Listening on http://{}", addr);

    // Run the server until it is stopped
    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }
}
